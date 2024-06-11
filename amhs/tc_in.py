from collections import defaultdict
import pandas as pd
import oracledb
import rediscluster as rds
import math
import networkx as nx
import threading
import json
import random
import copy

from mysql import connector
from loguru import logger as log
from contextlib import contextmanager
from algorithm.A_start.graph.srccode import *


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
    if p.mode == 1:
        p.db_pool = OracleConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn)
    else:
        p.db_pool = MysqlConnectionPool(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn,
                                        database=p.database)
    # old
    # p.db_connection = oracledb.connect(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn)
    # p.db_cursor = p.db_connection.cursor()

    # extracting local json documentation
    p = erect_map(p)

    # create static map
    # paths_static = p.Astart.more_path(p.map_info_unchanged)

    # p.all_stations = json.loads((open('all_station.json', 'r')).read())
    if p.pattern == 0:
        # vehicles info from Redis
        p = vehicle_load(p)
        # reading instructions
        p = read_instructions(p)
    return p


def erect_map(p):
    if p.Control().build_map:
        # read form db
        with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute('SELECT * FROM OHTC_MAP')
            df = pd.DataFrame(cursor.fetchall())
            p.original_map_info = df
            p.map_info_unchanged = DiGraph(p.status)
            p.Astart = AStart()
            for i in df.index:
                sp = df[1][i]  # start point
                ep = df[2][i]  # end point
                length = df[8][i]
                p.map_info_unchanged.add_node_from(sp, ep, length)
                p.map_info_unchanged.add_weighted_edges_from([(sp, ep, length)])
            # tensor generation
            # p.map_info_unchanged.create_matrix(df.values)
            db_conn.commit()
            cursor.close()
            track_generate_station(p, df)

        # added in 240603 by lby
        # a new model used to divide map into two pieces,
        # one for internal (within bays, nodes name started with 'W'),
        # another for external (high-ways, nodes name started with 'I'),
        # and there are links between two parts, including entrances('W-I') and outlets('I-W')
        p = map_divided(p)

        # old
        # p.db_cursor.execute('SELECT * FROM OHTC_MAP')
        # df = pd.DataFrame(p.db_cursor.fetchall())
        # df.to_json('./origin_map_info.json')
        # p.original_map_info = df
        # p = track_generate_station(p, df)  #table

        # building path map
        # p.map_info_unchanged = nx.DiGraph()
        # A* map
        # p.map_info_unchanged = DiGraph()
        # p.Astart = AStart()
        # for i in df.index:
        #     sp = df[1][i]  # start point
        #     ep = df[2][i]  # end point
        #     length = df[8][i]
        #     p.map_info_unchanged.add_node_from(sp,ep,length)
        #     p.map_info_unchanged.add_weighted_edges_from([(sp, ep, length)])

        # data = nx.readwrite.node_link_data(p.map_info)
        # f0 = open('weighted_adjacent_matrix.json', 'w')
        # f0.write(json.dumps(data))
        # f0.close()
    else:
        # load local map info
        p.stations_name = json.loads((open('stations_name.json', 'r')).read())
        data = json.loads((open('weighted_adjacent_matrix.json', 'r')).read())
        p.map_info_unchanged = nx.readwrite.node_link_graph(data)
        p.original_map_info = pd.read_json('./origin_map_info.json')
    return p


def map_divided(p):
    # this function is used to calculate paths in two parts of the map
    # input: map info from 'p'
    # output: routes in two parts
    df = p.original_map_info
    tmp = dict()
    tmp3, tmp4, all_bays = [], [], df[16].unique()
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
    # figuring out the shortest paths in every part
    paths_in_bays = dict()
    for k, v in tmp.items():
        pic = DiGraph(p.status)
        for k2, v2 in v['internal'].items():
            pic.add_weighted_edges_from([(k2[0], k2[1], v2)])
        pathA = dict(nx.all_pairs_dijkstra(pic, weight='weight'))
        paths_in_bays[k] = dict({"path": dict(pathA), "entrance": v['entrance'], "outlet": v['outlet']})
    p.internal_paths = paths_in_bays

    # paths_between_bays = dict()
    tmp5 = pd.DataFrame(data=math.inf, columns=all_bays, index=all_bays)
    for starting in tmp3:
        for ending in tmp4:
            if starting != ending:
                sb, eb = starting.split('-')[0], ending.split('-')[0]
                tmp6 = len(nx.shortest_path(p.map_info_unchanged, starting, ending))
                if tmp6 < tmp5[eb][sb]:
                    tmp5[eb][sb] = tmp6
                # paths_between_bays.update({(starting, ending): tmp5})
    # p.external_paths = paths_between_bays
    p.all_bays = all_bays
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
        p.redis_link = connection
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


def path_search(p, start, end, entrance, f_path, bayA, out):
    # path1
    # path1_end = p.internal_paths[bayA][out][0]  # todo:应该精确选取位置，这里随机录取
    path1_end = search_point(p,bayA,start,out)  # 精确选取位置
    path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])

    # path2
    bayB = end.split('-')[0]
    # path2_start = p.internal_paths[bayB][entrance][0]  # todo:应该精确选取位置，这里随机录取
    path2_start = search_point(p,bayB,entrance,end)  # 精确选取位置
    # path2 = nx.astar_path(p.map_info, path1_end, path2_start)
    path2 = nx.shortest_path(p.map_info, path1_end, path2_start)

    # path3
    path3 = copy.deepcopy(p.internal_paths[bayB][f_path][path2_start][1][end])
    return path1 + path2[1:-1] + path3

def search_point(p,bay,start,status,direction=1):
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
    order_list = []
    temp_cars = dict()
    for i in p.all_bays:
        temp_cars[i] = []
    number_task = len(p.taskList)
    if p.mode == 1:
        t0 = time.process_time()
        pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
        connection = rds.RedisCluster(connection_pool=pool)
        v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))
        log.info(f'cars number:{len(v)}, time:{time.process_time()-t0}')
        if v is None:
            return order_list
        all_vehicles_num = 0
        for value in v:
            tmp = vehicles_continue(p, value)
            if tmp[0]:
                continue

            # if value is None:
            #     continue
            # i = json.loads(value)
            # speed = int(i.get('currentSpeed'))
            # if i.get('ohtStatus_OnlineControl') != '1' or i.get('ohtStatus_ErrSet') != '0' or i.get(
            #         'ohtStatus_Idle') == '0' or speed == 0:
            #     continue
            # # 无法分配指令的车辆
            # s = int(i['position'])
            # idx = (p.original_map_info[3] < s) & (p.original_map_info[4] > s)
            # if float(p.original_map_info[4][idx].values[0] - s) / speed < p.tts:
            #     continue
            if len(order_list) >= number_task:
                return order_list
            i = json.loads(value)
            idx = tmp[1]
            flag = p.original_map_info[0][idx].values[0]
            bay = flag.split('-')[0]
            i['bay'] = bay
            i['mapId'] = flag
            temp_cars[bay].append(i)
            all_vehicles_num += 1
            if bay in p.bays_relation:
                task = p.bays_relation[bay]
                log.info(f"任务【{task[0].id}】+车辆【{i['ohtID']}】")
                if len(task) > 0:
                    order = task[0]
                    p.taskList.pop(p.taskList.index(order))
                    start = flag.split('_')[1]
                    end = p.all_stations[order.start_location]
                    taynum = p.stations_name[order.end_location]
                    path = copy.deepcopy(p.internal_paths[bay]['path'][start][1][end])
                    path.append(taynum)
                    order.vehicle_assigned = i.get('ohtID')
                    order.delivery_route = path
                    order_list.append(order)
                    # drop car and task
                    drop_car_task(temp_cars[i['bay']], i)
                    drop_car_task(task, task[0])
        log.info(f'本轮可用车辆数:{all_vehicles_num}')
        if all_vehicles_num == 0:
            return order_list

        if len(order_list) < number_task:
            try:
                for order in p.taskList:
                    car = near_bay_search(order.task_bay, p, temp_cars)
                    log.info(f"任务【{order.id}】+车辆【{car['ohtID']}】")
                    value = car('ohtID')
                    flag = car('mapId')
                    # temp_cars.pop(temp_cars.index(car))
                    out = 'outlet'
                    entrance = 'entrance'
                    f_path = 'path'
                    tay = flag.split('_')
                    bayA = tay[0].split('-')[0]
                    start = tay[1]
                    end = p.all_stations[order.start_location]
                    taynum = p.stations_name[order.end_location]
                    path = path_search(p, start, end, entrance, f_path, bayA, out)
                    path.append(taynum)
                    order.vehicle_assigned = value
                    order.delivery_route = path
                    order_list.append(order)
            except IndexError as e:
                log.error(e)
    # local running
    else:
        order_list = local_sql_search(p,order_list,[],number_task)
    return order_list
# 本地测试
def local_sql_search(p,order_list,temp_cars,number_task):
    with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute('''SELECT *
                FROM OHTC_CAR
                WHERE ohtID IS NOT NULL 
                AND (
                    (ohtStatus_OnlineControl <> '1' OR ohtStatus_ErrSet <> '0' OR ohtStatus_Idle = '0')
                    OR (ohtStatus_Roaming = '1' 
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Idle = '1') 
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncalling = '1')
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncall = '1'))
                    OR (ohtStatus_IsHaveFoup = '1' 
                        AND ohtStatus_MoveEnable = '1'
                        AND ohtStatus_Idle = '1')
                )
            ''')
    v = cursor.fetchall()
    for i in v:
        if len(order_list) > number_task:
            return order_list
        if not i:
            continue
        if i[28] != '1' or i[14] != '0' or i[17] == '0':
            continue
        bay = i[1]
        value = i[11]
        location = i[10]
        if bay in p.bays_relation:
            task = p.bays_relation[bay]
            if len(task) > 0:
                order = task[0]
                try:
                    speed = int(i[2])
                    if speed == 0:
                        speed = 1
                    number = float(p.original_map_info[p.original_map_info[0] == loaction][4].iloc[0])
                    ts = float(number - int(i[43])) / speed
                except:
                    ts = 0
                if ts < p.tts:
                    temp_cars.append(i)
                    continue
                else:
                    p.taskList.pop(p.taskList.index(order))
                    start = location.split('_')[1]
                    end = p.all_stations[order.start_location]
                    taynum = p.stations_name[order.end_location]
                    path = copy.deepcopy(p.internal_paths[bay]['path'][start][1][end])
                    path.append(taynum)
                    order.vehicle_assigned = value
                    order.delivery_route = path
                    order_list.append(order)
                    if len(task) > 0:
                        task.pop(0)
        else:
            temp_cars.append(i)
            pass
    if len(order_list) < number_task:
        try:
            for order in p.taskList:
                till = True
                while till:
                    car = random.choice(temp_cars)
                    loaction = car[10]
                    value = car[11]
                    speed = int(car[2])
                    if speed == 0:
                        speed = 1
                    number = float(p.original_map_info[p.original_map_info[0] == loaction][4].iloc[0])
                    ts = float(number - int(i[43])) / speed
                    if ts < p.tts:
                        temp_cars.pop(temp_cars.index(car))
                        continue
                    else:
                        temp_cars.pop(temp_cars.index(car))
                        out = 'outlet'
                        entrance = 'entrance'
                        f_path = 'path'
                        tay = location.split('_')
                        bayA = tay[0].split('-')[0]
                        start = tay[1]
                        end = p.all_stations[order.start_location]
                        taynum = p.stations_name[order.end_location]
                        # path1
                        path1_end = p.internal_paths[bayA][out][0]  # todo:应该精确选取位置，这里随机录取
                        path1 = copy.deepcopy(p.internal_paths[bayA][f_path][start][1][path1_end])

                        # path2
                        bayB = end.split('-')[0]
                        path2_start = p.internal_paths[bayB][entrance][0]  # todo:应该精确选取位置，这里随机录取
                        path2 = nx.shortest_path(p.map_info, path1_end, path2_start)

                        # path3
                        path3 = copy.deepcopy(p.internal_paths[bayB][f_path][path2_start][1][end])

                        # allpath
                        path = path1 + path2[1:-1] + path3

                        # return path
                        path.append(taynum)
                        order.vehicle_assigned = value
                        order.delivery_route = path
                        order_list.append(order)
                        till = False
        except:
            pass
    return order_list

def near_bay_search(bay0, p, cars):
    g = 0
    while 1:
        c = p.length_between_bays[bay0].sort_values()
        bay1 = c.index[0]
        if cars[bay1]:
            car = random.choice(cars[bay1])
            drop_car_task(cars[car['bay']], car)
            return car
        else:
            g += 1
        if g > p.max_search:
            return None


# static select car
def vehicle_load_static_fast(p):
    orederlist = []
    temcar = []
    if p.mode == 1:
        redis_pattern = """
                    WHERE ohtID IS NOT NULL
                    AND (
                        (ohtStatus_OnlineControl <> '1' OR ohtStatus_ErrSet <> '0')
                        OR (ohtStatus_Roaming = '1'
                            OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Idle = '1')
                            OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncalling = '1')
                            OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncall = '1'))
                        OR (ohtStatus_IsHaveFoup = '1'
                            AND ohtStatus_MoveEnable = '1'
                            AND ohtStatus_Idle = '1')
                    )
        """

        pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
        connection = rds.RedisCluster(connection_pool=pool)
        v = connection.mget(keys=connection.keys(pattern=f'{p.rds_search_pattern}{redis_pattern}*'))
        log.info('start a car search')

        for i in v:
            if len(orederlist) > 10:
                return orederlist
            if not i:
                continue
            ii = json.loads(i)
            bay = ii['bay']
            if bay in p.bays_relation:
                task = p.bays_relation[bay]
                if len(task) > 0:
                    start = ii['mapId'].split('_')[1]
                    orederlist.append(task[0])
                    task.pop(0)
            else:
                temcar.append(ii)
                pass
    else:
        with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute('''SELECT *
                FROM OHTC_CAR
                WHERE ohtID IS NOT NULL 
                AND (
                    (ohtStatus_OnlineControl <> '1' OR ohtStatus_ErrSet <> '0')
                    OR (ohtStatus_Roaming = '1' 
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Idle = '1') 
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncalling = '1')
                        OR (ohtStatus_MoveEnable = '1' AND ohtStatus_Oncall = '1'))
                    OR (ohtStatus_IsHaveFoup = '1' 
                        AND ohtStatus_MoveEnable = '1'
                        AND ohtStatus_Idle = '1')
                )
            ''')
        v = cursor.fetchall()
        for i in v:
            if len(orederlist) > 10:
                return orederlist
            if not i:
                continue
            bay = i[1]
            if bay in p.bays_relation:
                task = p.bays_relation[bay]
                if len(task) > 0:
                    order = task[0]
                    computeCarPath(order, i, p, orederlist, 1, 10, 11, True)
            else:
                temcar.append(i)
                pass
        if len(orederlist) < 10:
            for ty in p.taskList:
                till = True
                while till:
                    car = random.choice(temcar)
                    computeCarPath(ty, car, p, orederlist, 1, 10, 11, till)

    return orederlist


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


def track_generate(p):
    p.db_cursor.execute('SELECT * FROM OHTC_MAP')
    df = pd.DataFrame(p.db_cursor.fetchall())
    df.to_json('./origin_map_info.json')
    # searching shortest paths
    p = track_generate_shortest_path(p, df)
    # load station locations
    p = track_generate_station(p, df)
    return p


def track_generate_shortest_path(p, df):
    name_l = list(set(df[1].unique()) | set(df[2].unique()))
    adj = pd.DataFrame(0, columns=name_l, index=name_l)
    adj_valued = pd.DataFrame(0, columns=name_l, index=name_l)
    g = nx.DiGraph()
    for i in df.index:
        sp = df[1][i]  # start point
        ep = df[2][i]  # end point
        sl = df[3][i]
        el = df[4][i]
        length = df[8][i]
        adj[ep][sp] = 1
        adj_valued[ep][sp] = length
        # if sl > 0 and el > 0:
        g.add_weighted_edges_from([(sp, ep, length)])
    adj.to_json('./adjacent_matrix.json')
    sp = nx.all_pairs_bellman_ford_path(g)
    p.map_info = g
    all_path = dict()
    for i in sp:
        all_path[i[0]] = i[1]
    f0 = open('all_path.json', 'w')
    f0.write(json.dumps(all_path))
    f0.close()
    adj.to_json('./adjacent_matrix.json')
    adj_valued.to_json('./weighted_adjacent_matrix.json')

    p.all_routes = all_path
    p.adjacent_matrix = adj
    p.valued_adjacent_matrix = adj_valued
    return p


def track_generate_station(p, df):
    if p.all_stations is not None:
        return p
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
        p.all_stations = station_location
        p.stations_name = station_name
        db_conn.commit()
        cursor.close()
    # p.db_cursor.execute('SELECT * FROM OHTC_POSITION')
    # df2 = pd.DataFrame(p.db_cursor.fetchall())
    # station_location = dict()
    # station_name = dict()
    # for i in df2.index:
    #     num = df2[1][i]
    #     loc = df2[3][i]
    #     dft = df[(df[3] <= loc) & (df[4] >= loc)]
    #     station_location[num] = dft[1].values[0]
    #     station_name[num] = str(dft[3].values[0])
    # # 
    # # testing
    # # f1 = open('all_station.json', 'w')
    # # f1.write(json.dumps(station_location))
    # # f1.close()
    # # f2 = open('stations_name.json', 'w')
    # # f2.write(json.dumps(station_name))
    # # f2.close()
    # p.all_stations = station_location
    # p.stations_name = station_name
    return p

def track_generate_station_mulit(p, df):
    if p.all_stations is not None:
        return p

    df2 = get_station_data(p)

    station_location = dict()
    station_name = dict()

    num_threads = multiprocessing.cpu_count()
    newdata = split_data(df2, num_threads)

    set_station = partial(create_map_form_pd, orgine=df, start=station_location, end=station_name)
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

    # p.db_cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS=0 and VEHICLE='0'")
    # df = pd.DataFrame(p.db_cursor.fetchall())
    # debuger
    # if len(df) == 0:
    #     pass
    #     exit('All tasks have finished :)')
    # else:
    #     pass
    #     # print('Tasks number is', len(df))
    # n = 0
    # for i in df.index:
    #     tmp = p.Task()
    #     tmp.id = df[0][i]
    #     tmp.start_location = df[4][i]
    #     tmp.end_location = df[5][i]
    #     tmp.task_bay = tmp.start_location.split('_')[1]
    #     p.orders[tmp.id] = tmp
    #     n += 1
    #     if n >= p.Control().task_num:
    #         break
    # log.info(f'this is the task count:{n}')
    return p


def read_instructions_static(p):
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

    # p.db_cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS=0 and VEHICLE='0'")
    # df = pd.DataFrame(p.db_cursor.fetchall())
    # debuger
    # if len(df) == 0:
    #     pass
    #     exit('All tasks have finished :)')
    # else:
    #     pass
    #     # print('Tasks number is', len(df))
    # n = 0
    # for i in df.index:
    #     tmp = p.Task()
    #     tmp.id = df[0][i]
    #     tmp.start_location = df[4][i]
    #     tmp.end_location = df[5][i]
    #     tmp.task_bay = tmp.start_location.split('_')[1]
    #     p.orders[tmp.id] = tmp
    #     n += 1
    #     if n >= p.Control().task_num:
    #         break
    # log.info(f'this is the task count:{n}')
    return p


def linking_to_alg(p, p0):
    p.pattern = p0['circulating_on']
    p.oracle_user = p0['oracle_user']
    p.oracle_password = p0['oracle_password']
    p.oracle_dsn = p0['oracle_dsn']
    p.rds_connection = p0['rds_connection']
    p.rds_port = p0['rds_port']
    p.rds_search_pattern = p0['rds_search_pattern']
    return p
