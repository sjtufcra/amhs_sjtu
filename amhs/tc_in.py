import time
from collections import defaultdict
import pandas as pd
import oracledb
import redis.asyncio as rds
# import redislocal as db_redis
import redisclusterlocal as db_redis
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import asyncio

import math
import networkx as nx
import threading
import json
import random
import copy

from mysql import connector
from loguru import logger as log
from contextlib import contextmanager

from tc_out import *
from algorithm.A_start.graph.srccode import *


def generating(p):
    t0 = time.time()

    if p.mode == 1:
        p.db_pool = OracleConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn)
        p.db_redis = RedisConnectionPool(user=p.rds_connection, port=p.rds_port, cachekey=p.cache_key)
    else:
        p.db_pool = MysqlConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn,
                                        database=p.database)

    # load running status of algorithm
    # p = if_start(p)
    # if not p.algorithm_on:
    #     return p
    # from (erect_map)
    # split the map to 5 blocks, from A to E
    # each block contains 2 or 3 part of highways and several bays
    p = initiate_block(p)
    # read form the oracle database and build the graph of map
    p, tmp = map_build(p)
    # added in 240603 by lby
    # a new model used to divide map into two pieces,
    # one for internal (within bays, nodes name started with 'W'),
    # another for external (high-ways, nodes name started with 'I'),
    # and there are links between two parts, including entrances('W-I') and outlets('I-W')
    p = map_divided(p, tmp)
    # load all stations' information, but an old version
    if not p.debug_on:
        track_generate_station(p)
    log.info(f'generating success, time cost:{time.time() - t0}')
    return p


def initiate_block(p):
    t = p.block
    j = [('A', ['I001', 'I002'], [range(1, 35)]),
         ('B', ['I003', 'I004'], [range(35, 72)]),
         ('C', ['I004', 'I005'], [range(72, 106)]),
         ('D', ['I006', 'I007'], [range(106, 131)]),
         ('E', ['I003', 'I004', 'I005'], [131, 132]),
         ('F', ['W054', 'W055', 'W056'], [133, 134, 135])]
    for i in j:
        bays = []
        for k in i[2]:
            if isinstance(k, range):
                for k0 in k:
                    s = 'W' + str(k0).zfill(3)
                    bays.append(s)
            else:
                s = 'W' + str(k).zfill(3)
                bays.append(s)
        t[i[0]] = {'internal': dict(), 'external': dict(), 'neighbor': dict(),
                   'bays': bays, 'highways': i[1]}
    return p


def map_build(p):
    # read form db
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        cursor.execute('SELECT * FROM OHTC_MAP')
        df = pd.DataFrame(cursor.fetchall())
        p.original_map_info = df
        p.Astart = AStart()
        db_conn.commit()
        cursor.close()
    tmp = dict()
    tmp3, tmp4, all_bays = [], [], df[16].unique()
    p.all_bays = all_bays
    p.map_info_unchanged = DiGraph(p.status)
    if p.debug_on:
        p.map_info_unchanged.create_matrix(df.values)
    for i in all_bays:
        tmp[i] = {'internal': {}, 'entrance': [], 'outlet': []}
    for i in df.index:
        sp = df[1][i]  # start point
        ep = df[2][i]
        length = df[8][i]  # end point
        bay0 = sp.split('-')[0]
        bay1 = ep.split('-')[0]
        # if sp.startswith('W') and ep.startswith('W') and (bay0 == bay1):
        #     tmp[bay0]['internal'].update({(sp, ep): length})
        # elif sp.startswith('W') or ep.startswith('W'):
        #     if sp.startswith('W'):
        #         tmp[bay0]['outlet'].append(sp)
        #         tmp3.append(sp)
        #     if ep.startswith('W'):
        #         tmp[bay1]['entrance'].append(ep)
        #         tmp4.append(sp)
        if bay0 == bay1:
            tmp[bay0]['internal'].update({(sp, ep): length})
        else:
            tmp[bay0]['outlet'].append((sp, ep))
            tmp[bay1]['entrance'].append((sp, ep))
        # p.map_info_unchanged.add_node_from(sp, ep, length)
        p.map_info_unchanged.add_weighted_edges_from([(sp, ep, length)])
    return p, tmp


def map_divided(p, tmp):
    # figuring out the shortest paths in every part
    # indexed by a bay name and then by nodes both in this bay
    # example: r = p.block['A']['internal'][@bay_name][(@node1, @node2)]
    # r is a list containing a series of nodes, which is a viable route
    t0 = time.time()
    paths_in_bays = dict()
    for k, v in tmp.items():
        pic = DiGraph(p.status)
        for k2, v2 in v['internal'].items():
            pic.add_weighted_edges_from([(k2[0], k2[1], v2)])
        pathA = dict(nx.all_pairs_dijkstra(pic, weight='weight'))
        paths_in_bays[k] = dict({"path": dict(pathA), "entrance": v['entrance'], "outlet": v['outlet']})
        for k0, v0 in p.block.items():
            if k in v0['bays'] or k in v0['highways']:
                p.block[k0]['internal'].update({k: paths_in_bays[k]})
                # break
    # p.internal_paths = paths_in_bays
    log.info(f'time cost of internal routes:{time.time() - t0}')

    if p.debug_on:
        p.length_between_bays = pd.DataFrame(data=1, columns=p.all_bays, index=p.all_bays)
        return p

    # route within the different bays
    # indexed by 2 bay names and then by nodes in bays respectively
    # example: r = p.block['A']['external'][(@bay_name, @bay_name][(@node1, @node2)]
    t0 = time.time()
    e_matrix = pd.DataFrame(data=math.inf, columns=p.all_bays, index=p.all_bays)
    for k0, v0 in p.block.items():
        for i in v0['bays']:
            for j in v0['bays']:
                if i != j:
                    sp = v0['internal'][i]['outlet']
                    ep = v0['internal'][j]['entrance']
                    value0, path0 = math.inf, []
                    if sp and ep:
                        for i0 in sp:
                            for j0 in ep:
                                value, path = nx.bidirectional_dijkstra(p.map_info_unchanged, i0[0], j0[1])
                                if value < value0:
                                    path0 = path
                                    value0 = value
                    p.block[k0]['external'].update({(i, j): (value0, path0)})
                    e_matrix[j][i] = value0
        log.info(f'block {k0} completed')
    p.length_between_bays = e_matrix
    log.info(f'time cost of internal routes:{time.time() - t0}')

    # routes between highways and nodes inside bays
    # indexed by a bay name as well as a highway name, and then by nodes in this bay
    # example: r = p.block['A']['external'][(@highway, @bay_name][(@node1, @node2)]
    pass
    # sp in I001, ep in W001, both of them are in block 'A'
    # find the outlet track between them, which connected W001 with I001
    # assigning it to tmp, it's a tuple (outlet of W001, entrance of I001)
    # route0 = p.block['A']['internal'][W001][sp, tmp[0]]
    # route1 = p.block['A']['internal'][I001][tmp[1], ep]
    # the final route is equal to route0 + route1, and vise versa
    return p


# static select car
async def vehicle_load_static(p):
    temp_cars = dict()
    for i in p.all_bays:
        temp_cars[i] = []
    if p.mode == 1:
        t0 = time.time()
        # 异步调用
        asyncio.create_task(read_car_to_cache_back(p))
        cache = p.db_redis.get_cache()
        v = await cache.get(p.db_redis.cache_key)
        if v is None:
            # log.info(f'无车辆信息')
            return None
        log.info(f'cars number:{len(v)}, time:{time.time() - t0}')
        t1 = time.time()
        all_vehicles_num = 0
        c = [0, 0, 0, 0, 0]
        for value in v:
            tmp = vehicles_continue(p, value, c)
            if tmp[0]:
                continue
            i = json.loads(value)
            p.vehicles.update({i.get('ohtID'): i})
            try:
                flag = p.original_map_info[0][tmp[1]].values[0]
                bay = flag.split('-')[0]
                i['bay'] = bay
                i['mapId'] = flag
                temp_cars[bay].append(i)
                all_vehicles_num += 1
                if bay in p.bays_relation:
                    asyncio.create_task(assign_same_bay(p, bay, i, flag, temp_cars))
                if len(p.taskList) == 0:
                    return None
            except IndexError as e:
                log.error(f'error:{e},value:{tmp[1]},bay:{bay}')
                continue
        # log.info(f'phase 1 cost:{time.time()-t1}')

        t2 = time.time()
        log.info(f'本轮可用车辆数:{all_vehicles_num},'
                 f'空信息:{c[0]},车故障:{c[1]},未空闲:{c[2]},车速为0:{c[3]},离节点过近:{c[4]}')
        if all_vehicles_num == 0:
            return None
        try:
            for order in p.taskList:
                # car = near_bay_search(order.task_bay, p, temp_cars)
                car = near_bay_search_new(order, p, temp_cars)
                if not car:
                    continue
                value = car.get('ohtID')
                # log.info(f"任务:{order.id}+车辆:{value}")
                flag = car.get('mapId')
                # temp_cars.pop(temp_cars.index(car))
                out = 'outlet'
                entrance = 'entrance'
                f_path = 'path'
                # tay = flag.split('_')
                # bayA = tay[0].split('-')[0]
                # when vehicle is located in the connected track, like "I001-01, I002-33"
                # its real bay should be the I002
                bayA = flag.split('_')[1].split('-')[0]
                start = get_start(car, p, flag)
                # # 同步获取
                # path = path_search(p, start, entrance, f_path, bayA, out, order)
                path = path_search_new(p, start, entrance, f_path, bayA, out, order)
                if path is None:
                    continue
                order.vehicle_assigned = value
                order.delivery_route = path
                asyncio.create_task(output_new(p, order))
        except IndexError as e:
            log.error(e)
        log.info(f'phase 2 cost:{time.time() - t2}')
        log.info(f'phase 1 and 2 cost:{time.time() - t1}')
    return None


def get_start(car, p, tay):
    # 同步获取
    ipx = car.get('ohtIP')
    num = car.get('ohtID')[1:]
    car_key = f'Car:location:{ipx}_1{num}'
    try:
        start = get_redis_position(p, car_key)
    except ValueError as e:
        start = tay.split('_')[1]
    return start


def get_redis_position(p, key):
    try:
        # t0 = time.time()
        redis = p.db_redis.get_redis()
        jsonData = redis.get(key)
        value = json.loads(jsonData).get('mapId')
        point = value.split('_')[1]
        # end = time.time() - t0
        # print(f'all_time:{end}')
        return point
    except ValueError as e:
        log.error(f'error:{e},key:{key}')
        return None


def vehicle_load(p):
    # load from 'redislocal'
    # 初始化
    p.vehicles_bay_get = clear_data()
    p.vehicles_bay_send = clear_data()
    p.vehicles_get = dict()
    p.vehicles_send = dict()
    if p.mode == 1:
        pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
        connection = rds.RedisCluster(connection_pool=pool)
        v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))
        # vehicles = dict()
        log.info('start a car search')
        for i in v:
            if not i:
                continue
            ii = json.loads(i)
            if ii.get('ohtID'):
                if ii.get('ohtStatus_OnlineControl') != '1' or ii.get('ohtStatus_ErrSet') != '0':
                    continue
                if ((ii.get('ohtStatus_Roaming') == '1' or (
                        ii.get('ohtStatus_MoveEnable') == '1' and ii.get('ohtStatus_Idle') == '1')) or (
                            ii.get('ohtStatus_MoveEnable') == '1' and ii.get('ohtStatus_Oncalling') == '1')) or (
                        ii.get('ohtStatus_MoveEnable') == '1' and ii.get('ohtStatus_Oncall') == '1'):
                    p.vehicles_get[ii.get('ohtID')] = ii
                    set_data(p.vehicles_bay_get, ii["bay"], ii)
                else:
                    if ii.get('ohtStatus_IsHaveFoup') == '1' and (
                            ii.get('ohtStatus_MoveEnable') == '1' and ii.get('ohtStatus_Idle') == '1'):
                        p.vehicles_send[ii.get('ohtID')] = ii
                        set_data(p.vehicles_bay_send, ii["bay"], ii)
                    else:
                        continue
    else:
        with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute('SELECT * FROM OHTC_CAR')
        v = cursor.fetchall()
        for i in v:
            if not i:
                continue
            value = i[11]
            if value:
                if i[28] != '1' or i[14] != '0':
                    continue
                if ((i[35] == '1' or (i[24] == '1' and i[17] == '1')) or (i[24] == '1' and i[27] == '1')) or (
                        i[24] == '1' and i[26] == '1'):
                    p.vehicles_get[value] = i
                    set_data(p.vehicles_bay_get, i[1], i)
                else:
                    if i[18] == '1' and (i[24] == '1' and i[17] == '1'):
                        p.vehicles_send[value] = i
                        set_data(p.vehicles_bay_send, i[1], i)
                    else:
                        continue

            # old code
            # mapID('10')是起点：[sp, ep] = ii['mapId'].split('_')[1] or [sp, ep] = ii[10].split('_')[1]
            # vehicles[ii['ohtID']] = ii
            # existing vehicles in one path
            # if not ii.get('mapId'):
            #     continue
            # [sp, ep] = ii['mapId'].split('_')
            # if not p.vehicle_jam.get((sp, ep)):
            #     p.vehicle_jam[sp, ep] = 1
            # else:
            #     p.vehicle_jam[sp, ep] += 1
    # log.info(f'Available vehicles number is:{ len(vehicles)}')
    # p.vehicles = vehicles
    return p


def vehicles_continue(p, i, c):
    if i is None:
        c[0] += 1
        return True, None
    i = json.loads(i)
    if i.get('ohtStatus_OnlineControl') != '1' or i.get('ohtStatus_ErrSet') != '0':
        c[1] += 1
        return True, None
    if i.get('ohtStatus_Idle') == '1':
        c[2] += 1
        return True, None
    speed = int(i.get('currentSpeed'))
    if speed == 0:
        c[3] += 1
        return True, None
    # 无法分配指令的车辆
    s = int(i['position'])
    try:
        idx = (p.original_map_info[3] <= s) & (p.original_map_info[4] >= s)
        return False, idx

        # todo: 判断是否过近
        # if float(p.original_map_info[4][idx].values[0] - s) / speed < p.tts:
        #     c[4] += 1
        #     return True, None
        # else:
        #     return False, idx
    except Exception as e:
        log.error(f'erro:{e},value:{p.original_map_info[4]},data:{p.original_map_info[4][idx]}')
        return True, None


def drop_car_task(x, i):
    x.pop(x.index(i))
    return 0


# new
def path_search_new(p, start, entrance, f_path, bayA, out, order):
    try:
        tmp = p.block[order.block_location]['internal']
        end = p.all_stations.get(order.start_location)
        # path1
        path1_end = search_point_new(tmp, bayA, start, out)  # 精确选取位置
        path1 = get_path_from_bay(p, tmp, bayA, f_path, start, path1_end)

        # path2
        bayB = end.split('-')[0]
        path2_start = search_point_new(tmp, bayB, end, entrance, direction=0)  # 精确选取位置
        path2 = nx.shortest_path(p.map_info, path1_end, path2_start)

        # path3
        path3 = get_path_from_bay(p, tmp, bayB, f_path, path2_start, end)
        path = path1 + path2[1:-1] + path3
        # path.append(p.stations_name.get(end))
        # return path
    except Exception as e:
        # log.error(f'path_search_new error:{e}')
        end = p.all_stations.get(order.start_location)
        path = nx.shortest_path(p.map_info, start, end)
        # path.append(p.stations_name.get(end))
        log.warning(f'path_search_new error:{e},continue this task')
    path.append(p.stations_name.get(order.start_location))
    return path


def get_path_from_bay(p, tmp, bay, f_path, start, end):
    try:
        plist = tmp[bay][f_path][start][1].get(end)
        path = copy.deepcopy(plist)
        return path
    except IndexError as e:
        log.warning(f'warning:{e},value is None')
        # todo: 可以重新计算
        # path = nx.shortest_path(p.map_info, start, end)
        # return path
        return None


# old
def path_search(p, start, entrance, f_path, bayA, out, order):
    end = p.all_stations[order.start_location]
    # path1
    # path1_end = p.internal_paths[bayA][out][0]  # todo:应该精确选取位置，这里随机录取
    path1_end = search_point(p, bayA, start, out)  # 精确选取位置
    # path2
    bayB = end.split('-')[0]
    # path2_start = p.internal_paths[bayB][entrance][0]  # todo:应该精确选取位置，这里随机录取
    path2_start = search_point(p, bayB, end, entrance, direction=0)  # 精确选取位置
    # path2 = nx.astar_path(p.map_info, path1_end, path2_start)
    if path1_end is None or path2_start is None:
        return None

    path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])
    path2 = nx.shortest_path(p.map_info, path1_end, path2_start)

    path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])
    path2 = nx.shortest_path(p.map_info, path1_end, path2_start)
    path3 = copy.deepcopy(p.internal_paths[bayB][f_path][path2_start][1][end])
    return path1 + path2[1:-1] + path3


def search_point(p, bay, start, status, direction=1):
    try:
        txt = p.internal_paths[bay][status]
        if len(txt) == 0:
            log.warning(f'bay:{bay},point:{start},status:{status},data:{p.internal_paths[bay]}')
            return None

        pointA, pointB = p.internal_paths[bay][status]
        path = p.internal_paths[bay]['path']
        if direction:
            # 出口
            tagA = path[start][0][pointA]
            tagB = path[start][0][pointB]
        else:
            # 入口
            tagA = path[pointA][0][start]
            tagB = path[pointB][0][start]
        return pointB if tagA >= tagB else pointA
    except ValueError as e:
        log.error(f"error:{e},value:{p.internal_paths[bay]}")
        return None


def search_point_new(tmp0, bay, start, status, direction=1):
    try:
        tmp1 = tmp0[bay][status]
        path = tmp0[bay]['path']
        if direction:
            # 出口
            pointA = tmp1[0][0]
            pointB = tmp1[1][0]
            tagA = path[start][0].get(pointA, float('inf'))
            tagB = path[start][0].get(pointB, float('inf'))
        else:
            # 入口
            pointA = tmp1[0][1]
            pointB = tmp1[1][1]
            tagA = path[pointA][0].get(start, float('inf'))
            tagB = path[pointB][0].get(start, float('inf'))
        return pointB if tagA >= tagB else pointA
    except ValueError as e:
        log.error(f"error:{e},value:{tmp0[bay]},status:{status}")
        return None


async def assign_same_bay(p, bay, i, flag, temp_cars):
    task = p.bays_relation[bay]
    tmp_id = i.get('ohtID')
    if len(task) > 0:
        order = task[0]
        end_station = order.start_location
        try:
            # log.info(f"任务:{order.id},车辆:{tmp_id}")
            p.taskList.pop(p.taskList.index(order))
            # tay = flag.split('_')[1]
            start = get_start(i, p, flag)
            # # 同步获取
            # idx = 'ohtIP'
            # num = i.get('ohtID')[1:]
            # car_key = f'Car:location:{i.get(idx)}_1{num}'
            # start = get_redis_position(p, car_key)
            # if start is None:
            #     start = flag.split('_')[1]

            end = p.all_stations.get(end_station)
            # path = copy.deepcopy(p.internal_paths[bay]['path'][start][1][end])

            path = internal_path_search(start, end, bay, p)
            path.append(p.stations_name.get(end_station))

            order.vehicle_assigned = tmp_id
            order.delivery_route = path
            await output_new(p, order)
            # drop car and task
            drop_car_task(temp_cars.get(bay), i)
            drop_car_task(task, task[0])
        except Exception as e:
            start = flag.split('_')[1]
            end = p.all_stations.get(end_station)
            path = nx.shortest_path(p.map_info, start, end)
            path.append(p.stations_name.get(end_station))
            order.vehicle_assigned = tmp_id
            order.delivery_route = path
            await output_new(p, order)
            # todo:部分数据不在p.block['C']['internal'][bay]['path'][start][1]
            log.warning(f"warning:{e},this a computing path")
            return
    return 0


def internal_path_search(start, end, bay, p):
    for k, v in p.block.items():
        if bay in v['bays']:
            try:
                path = v['internal'][bay]['path'][start][1][end]
            except IndexError as e:
                path = nx.shortest_path(p.map_info, start, end)
            return path


def near_bay_search_new(order, p, cars):
    # search the left and right bay, sometimes only one bay can be searched
    block_tmp = order.block_location
    bay0 = int(order.task_bay[1:])
    bay1 = 'W' + str(bay0 + 1).zfill(3)
    bay2 = 'W' + str(bay0 - 1).zfill(3)
    candidate_bay = []
    if bay1 in p.block[block_tmp]['bays']:
        candidate_bay.append(bay1)
    if bay2 in p.block[block_tmp]['bays']:
        candidate_bay.append(bay2)
    for i in candidate_bay:
        car_tmp = cars.get(i)
        if car_tmp:
            car = random.choice(car_tmp)
            drop_car_task(car_tmp, car)
            # todo:车辆正处于出口、入口时，可不予分配，应当继续搜索
            # if location_of_car(car):
            #     continue
            return car
    # if no cars nearby can be used, searching the cars on highways randomly
    for i in p.block[block_tmp]['highways']:
        car_tmp = cars.get(i)
        if car_tmp:
            car = random.choice(car_tmp)
            drop_car_task(car_tmp, car)
            # todo:车辆正处于出口、入口时，可不予分配，应当继续搜索
            # if location_of_car(car):
            #     continue
            return car
    return None


def location_of_car(car):
    track = car.get('mapId').split('_')
    n0 = track[0].split('-')[0]
    n1 = track[1].split('-')[0]
    if n0 != n1:
        return True
    else:
        return False


def near_bay_search(bay0, p, cars):
    g = 0
    while 1:
        try:
            c = p.length_between_bays[bay0].sort_values()
            bay1 = c.index[0]
            car_tmp = cars.get(bay1)
            if car_tmp:
                car = random.choice(car_tmp)
                drop_car_task(car_tmp, car)
                return car
            else:
                g += 1
            if g > p.max_search:
                return None
        except Exception as e:
            log.error(f"error:{e},bay:{bay0},p:{p.length_between_bays}")
            return None


# 异步设置缓存函数
async def read_car_to_cache_back(p):
    data = await cache_redis(p)
    cache = p.db_redis.get_cache()
    await cache.set(p.db_redis.cache_key, data)


# 异步读取redis缓存
async def cache_redis(p):
    redis = p.db_redis.get_connection()
    keys = await redis.keys(pattern=p.rds_search_pattern)
    values = list()
    for key in keys:
        value = await redis.get(key)
        values.append(value)
    return values


# 同步函数
# def read_car_tmp(p):
#     pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
#     connection = rds.RedisCluster(connection_pool=pool)
#     v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))
#     p.vehicles_get = v

def computeCarPath(ty, car, p, orderlist, inx, flag, mapid, plist=False, boll=False, ):
    bay = car[inx]
    order = ty
    loaction = car[flag]
    ts = (p.original_map_info[p.original_map_info[0] == loaction][4] - int(car[43])) / int(car[2]).values[0]
    if ts < p.tts:
        boll = True
        pass
    else:
        start = car[flag].split('_')[1]
        end = p.all_stations[bay]
        taynum = p.stations_name[bay]
        path = p.internal_paths[bay][start][end].append(taynum)

        if plist:
            p.taskList.pop(p.taskList.index(order))

        order.vehicle_assigned = car[mapid]
        order.delivery_route = path
        orderlist.append(order)
        ty.pop(0)

        boll = False


# set data in element
def set_data(dictionary, key, data):
    # 增加输入校验
    if not isinstance(dictionary, dict):
        log.error("dictionary must be a dictionary")
    if key is not None and not isinstance(key, str):
        log.error("key must be a string or None")

    # 简化代码逻辑，同时处理 key 为 None 和非 None 的情况
    dictionary[key if key is not None else 'default'] += [data]


# clear data in element
def clear_data():
    return defaultdict(list, dict())


def clear_dict():
    return dict()


def track_generate_station(p):
    df = p.original_map_info
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        cursor.execute('SELECT * FROM OHTC_POSITION')
        df2 = pd.DataFrame(cursor.fetchall())
        station_location = dict()
        station_name = dict()
        for i in df2.index:
            num = df2[1][i]
            loc = df2[3][i]
            dft = df[(df[3] <= loc) & (df[4] >= loc)]
            station_location[num] = dft[1].values[0]  # 台位所在的轨道起点编号
            station_name[num] = df2[2][i]  # 台位所在轨道的台位编号
        p.all_stations.update(station_location)
        p.stations_name.update(station_name)
        db_conn.commit()
        cursor.close()
    return p


def track_generate_station_mulit(p, df):
    if p.all_stations is not None:
        return p

    df2 = get_station_data(p)

    station_location = dict()
    station_name = dict()

    num_threads = multiprocessing.cpu_count()  # type: ignore
    newdata = split_data(df2, num_threads)

    set_station = partial(create_map_form_pd, orgine=df, start=station_location, end=station_name)  # type: ignore
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pross:
        results = pross.map(set_station, newdata)

    for start, end in results:
        station_location.update(start)
        station_name.update(end)

    p.all_stations = station_location
    p.stations_name = station_name

    return p


# 多进程处理台位数据
def create_map_form_pd(iteram, orgine, start, end):
    for i in iteram.index:
        num = iteram[1][i]
        loc = iteram[3][i]
        dft = orgine[(orgine[3] <= loc) & (orgine[4] >= loc)]
        if not dft.empty:
            start[num] = dft[1].values[0]
            end[num] = str(dft[3].values[0])
    return start, end


# 台位数据分割
def split_data(data, num_processes):
    size = len(data)
    split_size = size // num_processes
    return [data[i:i + split_size] for i in range(0, size, split_size)]


# 读取数据库数据
def get_station_data(p):
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        cursor.execute('SELECT * FROM OHTC_POSITION')
        df2 = pd.DataFrame(cursor.fetchall())
        cursor.close()
        db_conn.commit()
    return df2


def read_instructions(p):
    # oracle
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS IN (0,10)  and VEHICLE='0'")
        df = pd.DataFrame(cursor.fetchall())
        db_conn.commit()
        cursor.close()
    n = 0
    for i in df.index:
        tmp = p.Task()
        tmp.id = df[0][i]
        tmp.start_location = df[4][i]
        tmp.end_location = df[5][i]
        tmp.task_bay = tmp.start_location.split('_')[1]
        p.orders[tmp.id] = tmp
        n += 1
        if n >= p.task_num:
            break
    log.info(f'this is the task count:{n}')
    return p


def read_instructions_static(p):
    # log.info(f"开始读取任务")
    # oracle
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        sql = "SELECT * FROM TRANSFER_TABLE WHERE VEHICLE='0'"
        cursor.execute(sql)
        df = pd.DataFrame(cursor.fetchall())
        # log.info(f'task count:{len(df)}')
        db_conn.commit()
        cursor.close()
    n = 0
    p.bays_relation = clear_data()
    p.taskList = []
    for i in df.index:
        tmp = p.Task()
        tmp.id = df[0][i]
        tmp.start_location = df[4][i]
        tmp.end_location = df[5][i]
        tmp.task_bay = tmp.start_location.split('_')[1]
        # added in 240612 by zq
        for k, v in p.block.items():
            if tmp.task_bay in v['bays']:
                tmp.block_location = k
                break
        set_data(p.bays_relation, tmp.task_bay, tmp)
        p.taskList.append(tmp)
        n += 1
        if n >= p.task_num:
            break
    # added in 240621, to count numbers of being read and finished
    p.tasks_finish_count = 0
    p.tasks_load_count = n
    # log.info(f'本轮任务数:{n}')
    return p


# server config


# server config
class OracleConnectionPool:
    def __init__(self, dsn, user, password, max_connections=5):
        self.dsn = dsn
        self.user = user
        self.password = password
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()

    @contextmanager
    def get_connection(self):
        with self.lock:
            while len(self.connections) < self.max_connections:
                conn = oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
                self.connections.append(conn)
            conn = self.connections.pop(0)
        try:
            yield conn
        finally:
            with self.lock:
                self.connections.append(conn)


# local config
class MysqlConnectionPool:
    def __init__(self, dsn, user, password, database, max_connections=5):
        self.host = dsn
        self.user = user
        self.database = database
        self.password = password
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()

    @contextmanager
    def get_connection(self):
        with self.lock:
            while len(self.connections) < self.max_connections:
                conn = connector.connect(user=self.user, password=self.password, host=self.host, database=self.database)
                self.connections.append(conn)
            conn = self.connections.pop(0)
        try:
            yield conn
        finally:
            with self.lock:
                self.connections.append(conn)


class RedisConnectionPool:
    def __init__(self, user, port, password=None, cachekey='', max_connections=5):
        self.host = user
        self.port = port
        self.cache_key = cachekey
        self.password = password
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        self.reds = None
        self.cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

    def get_connection(self):
        return self.reds

    def get_cache(self):
        return self.cache

    async def initialize_redis(self):
        self.reds = rds.from_url(f'redis://{self.host}:{self.port}', decode_responses=True)
        pool = db_redis.ClusterConnectionPool(host=self.host, port=self.port)
        self.redis = db_redis.RedisCluster(connection_pool=pool)

    def get_redis(self):
        return self.redis
