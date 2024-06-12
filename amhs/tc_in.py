import time
from collections import defaultdict
import pandas as pd
import oracledb
import rediscluster as rds
# import aioredis as rds
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
from .tc_out import *
from .algorithm.A_start.graph.srccode import *



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


def generating(p):
    t0 = time.time()
    if p.mode == 1:
        p.db_pool = OracleConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn)
    else:
        p.db_pool = MysqlConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn,
                                        database=p.database)
    # extracting local json documentation
    p = erect_map(p)
    log.info(f'generating success, time cost:{time.time() - t0}')
    return p


def erect_map(p):
    if p.Control().build_map:
        # read form db
        with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute('SELECT * FROM OHTC_MAP')
            df = pd.DataFrame(cursor.fetchall())
            p.original_map_info = df
            p.Astart = AStart()
            db_conn.commit()
            cursor.close()
        # load all stations' information, but an old version
        if not p.debug_on:            
            track_generate_station(p, df)
        # added in 240603 by lby
        # a new model used to divide map into two pieces,
        # one for internal (within bays, nodes name started with 'W'),
        # another for external (high-ways, nodes name started with 'I'),
        # and there are links between two parts, including entrances('W-I') and outlets('I-W')
        p = map_divided(p)
    return p


def map_divided(p):
    # this function is used to calculate paths in two parts of the map
    # input: map info from 'p'
    # output: routes in two parts
    df = p.original_map_info
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
        if sp.startswith('W') and ep.startswith('W') and (bay0 == bay1):
            tmp[bay0]['internal'].update({(sp, ep): length})
        elif sp.startswith('W') or ep.startswith('W'):
            if sp.startswith('W'):
                tmp[bay0]['outlet'].append(sp)
                tmp3.append(sp)
            if ep.startswith('W'):
                tmp[bay1]['entrance'].append(ep)
                tmp4.append(sp)
        p.map_info_unchanged.add_node_from(sp, ep, length)
        p.map_info_unchanged.add_weighted_edges_from([(sp, ep, length)])
        # flyd(p.map_info_unchanged)

    t0 = time.time()
    # figuring out the shortest paths in every part
    paths_in_bays = dict()
    for k, v in tmp.items():
        pic = DiGraph(p.status)
        for k2, v2 in v['internal'].items():
            pic.add_weighted_edges_from([(k2[0], k2[1], v2)])
        pathA = dict(nx.all_pairs_dijkstra(pic, weight='weight'))
        paths_in_bays[k] = dict({"path": dict(pathA), "entrance": v['entrance'], "outlet": v['outlet']})
    p.internal_paths = paths_in_bays
    log.info(f'time cost of internal routes:{time.time() - t0}')
    t0 = time.time()
    # paths_between_bays = dict()
    tmp5 = pd.DataFrame(data=math.inf, columns=all_bays, index=all_bays)
    for start in tmp3:
        for end in tmp4:
            if start != end:
                sb, eb = start.split('-')[0], end.split('-')[0]
                # A*
                # starting = p.map_info_unchanged.get_node_by_id(start)
                # ending = p.map_info_unchanged.get_node_by_id(end)
                # p.map_info_unchanged.set_start_and_goal(starting, ending)
                # path0 = p.Astart.a_star_search(p.map_info_unchanged)
                # nx
                # tmp6 = nx.dijkstra_path_length(p.map_info_unchanged, source=start, target=end)
                tmp6 = nx.shortest_path_length(p.map_info_unchanged, source=start, target=end)
                if tmp6 < tmp5[eb][sb]:
                    tmp5[eb][sb] = tmp6
                # paths_between_bays.update({(starting, ending): tmp5})
    log.info(f'time cost of external routes:{time.time() - t0}')
    # p.external_paths = paths_between_bays
    p.length_between_bays = tmp5
    return p


def vehicle_load(p):
    # load from 'redis'
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


def vehicles_continue(p, i):
    if i is None:
        return True, None
    i = json.loads(i)
    speed = int(i.get('currentSpeed'))
    if i.get('ohtStatus_OnlineControl') != '1' or i.get('ohtStatus_ErrSet') != '0' or i.get(
            'ohtStatus_Idle') == '0' or speed == 0:
        return True, None
    # 无法分配指令的车辆
    s = int(i['position'])
    idx = (p.original_map_info[3] < s) & (p.original_map_info[4] > s)
    if float(p.original_map_info[4][idx].values[0] - s) / speed < p.tts:
        return True, None
    else:
        return False, idx


def drop_car_task(x, i):
    x.pop(x.index(i))
    return 0


def path_search(p, start, entrance, f_path, bayA, out, order):
    end = p.all_stations[order.start_location]
    # path1
    # path1_end = p.internal_paths[bayA][out][0]  # todo:应该精确选取位置，这里随机录取
    path1_end = search_point(p,bayA,start,out)  # 精确选取位置
    # path2
    bayB = end.split('-')[0]
    # path2_start = p.internal_paths[bayB][entrance][0]  # todo:应该精确选取位置，这里随机录取
    path2_start = search_point(p,bayB,end,entrance,direction=0)  # 精确选取位置
    # path2 = nx.astar_path(p.map_info, path1_end, path2_start)
    if path1_end is None or path2_start is None:
        return None

    path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])
    path2 = nx.shortest_path(p.map_info, path1_end, path2_start)

    path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])
    path2 = nx.shortest_path(p.map_info, path1_end, path2_start)
    path3 = copy.deepcopy(p.internal_paths[bayB][f_path][path2_start][1][end])
    return path1 + path2[1:-1] + path3

def search_point(p,bay,start,status,direction=1):
    log.warning(f'bay:{bay},point:{start},status:{status}')
    txt = p.internal_paths[bay][status]
    if len(txt)==0:
        return None
    pointA,pointB = p.internal_paths[bay][status]
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

# static select car
def vehicle_load_static(p):
    t_start = time.time()
    log.info(f"开始分配任务")
    temp_cars = dict()
    for i in p.all_bays:
        temp_cars[i] = []
    if p.mode == 1:
        t0 = time.time()
        # 同步调用
        # pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
        # connection = rds.RedisCluster(connection_pool=pool)
        # v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))
        # 异步调用
        asyncio.run(read_car_to_cach(p))
        log.info(f'cars number:{len(p.vehicles_get)}, time:{time.time()-t0}')
        if p.vehicles_get is None:
            return None
        all_vehicles_num = 0
        t1 = time.time()
        for value in p.vehicles_get:
            tmp = vehicles_continue(p, value)
            if tmp[0]:
                continue
            i = json.loads(value)
            flag = p.original_map_info[0][tmp[1]].values[0]
            bay = flag.split('-')[0]
            i['bay'] = bay
            i['mapId'] = flag
            temp_cars[bay].append(i)
            all_vehicles_num += 1
            if bay in p.bays_relation:
                assign_same_bay(p, bay, i, flag, temp_cars)
            if len(p.taskList) == 0:
                return None
        log.info(f'phase 1 cost:{time.time()-t1}')
        
        t2 = time.time()
        log.info(f'本轮可用车辆数:{all_vehicles_num}')
        if all_vehicles_num == 0:
            return None
        try:
            for order in p.taskList:
                car = near_bay_search(order.task_bay, p, temp_cars)
                if not car:
                    continue
                value = car.get('ohtID')
                log.info(f"任务:{order.id}+车辆:{value}")
                flag = car.get('mapId')
                # temp_cars.pop(temp_cars.index(car))
                out = 'outlet'
                entrance = 'entrance'
                f_path = 'path'
                tay = flag.split('_')
                bayA = tay[0].split('-')[0]
                start = tay[1]
                path = path_search(p, start, entrance, f_path, bayA, out, order)
                order.vehicle_assigned = value
                order.delivery_route = path
                output_new(p, order)
        except IndexError as e:
            log.error(e)
        log.info(f'phase 2 cost:{time.time() - t2}')
        log.info(f'phase 1 and 2 cost:{time.time() - t1}')
    return None


def assign_same_bay(p, bay, i, flag, temp_cars):
    task = p.bays_relation[bay]
    tmp_id = i.get('ohtID')
    if len(task) > 0:
        order = task[0]
        log.info(f"任务:{order.id},车辆:{tmp_id}")
        p.taskList.pop(p.taskList.index(order))
        end_station = order.start_location
        start = flag.split('_')[1]
        end = p.all_stations.get(end_station)
        path = copy.deepcopy(p.internal_paths[bay]['path'][start][1][end])
        path.append(p.stations_name.get(end_station))

        order.vehicle_assigned = tmp_id
        order.delivery_route = path
        output_new(p, order)
        # drop car and task
        drop_car_task(temp_cars.get(bay), i)
        drop_car_task(task, task[0])
    return 0

def near_bay_search(bay0, p, cars):
    g = 0
    while 1:
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
# 异步线程
async def read_car_to_cache_async(p):
    redis = rds.from_url(
        f'redis://{p.rds_connection}:{p.rds_port}/',
        decode_responses=True
    )
    try:
        keys = await redis.keys(pattern=p.rds_search_pattern)
        values = await redis.mget(keys)
        p.vehicles_get = values
    finally:
        await redis.close()
async def read_car_to_cach(p):
    # 异步读取
    await read_car_to_cache_back(p)
    # # btkread_car_to_cache_back(p)
    # if p.vehicles_get is None:
    #     pass
    # # else:
    #     await read_car_to_cache_async(p)

# 异步
async def read_car_to_cache_back(p):
    pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
    connection = rds.RedisCluster(connection_pool=pool)
    v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))
    p.vehicles_get = v 
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


def track_generate_station(p, df):
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
            station_location[num] = dft[1].values[0] #台位所在的轨道起点编号
            station_name[num] = df2[2][i]#台位所在轨道的台位编号
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

    num_threads = multiprocessing.cpu_count() # type: ignore
    newdata = split_data(df2, num_threads)

    set_station = partial(create_map_form_pd, orgine=df, start=station_location, end=station_name) # type: ignore
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pross:
        results = pross.map(set_station, newdata)

    for start, end in results:
        station_location.update(start)
        station_name.update(end)

    p.all_stations = station_location
    p.stations_name = station_name

    return p

# 多进程处理台位数据
def create_map_form_pd(iteram, orgine,start,end):
    for i in iteram.index:
        num = iteram[1][i]
        loc = iteram[3][i]
        dft = orgine[(orgine[3] <= loc) & (orgine[4] >= loc)]
        if not dft.empty:
            start[num] = dft[1].values[0]
            end[num] = str(dft[3].values[0])
    return start,end

# 台位数据分割
def split_data(data,num_processes):
    size = len(data)
    split_size = size//num_processes
    return [data[i:i+split_size]for i in range(0,size,split_size)]

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
    log.info(f"开始读取任务")
    # oracle
    with p.db_pool.get_connection() as db_conn:
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS in (0,10) or VEHICLE='0'")
        df = pd.DataFrame(cursor.fetchall())
        log.info(f'task count:{len(df)}')
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
        set_data(p.bays_relation, tmp.task_bay, tmp)
        p.taskList.append(tmp)
        n += 1
        if n >= p.task_num:
            break
    log.info(f'this is the task count:{n}')
    return p

