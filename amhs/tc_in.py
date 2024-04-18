import pandas as pd
import oracledb
from loguru import logger as log
import rediscluster as rds
import networkx as nx
import json
from .algorithm.A_start.graph.srccode import AStar, Node, Edge,Graph

def generating(p):
    p.db_connection = oracledb.connect(user=p.oracle_user, password=p.oracle_password, dsn=p.oracle_dsn)
    p.db_cursor = p.db_connection.cursor()

    # extracting local json documentation
    p = erect_map(p)
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
        p.db_cursor.execute('SELECT * FROM OHTC_MAP')
        df = pd.DataFrame(p.db_cursor.fetchall())
        # df.to_json('./origin_map_info.json')
        p.original_map_info = df
        p = track_generate_station(p, df) #台位信息10w+
        # building path map
        p.map_info_unchanged = nx.DiGraph()
        # A*算法
        p.map_info_unchanged = Graph()
        for i in df.index:
            sp = df[1][i]  # start point
            ep = df[2][i]  # end point
            length = df[8][i]
            #A* 算法
            p.map_info_unchanged.add_node_from(sp,ep,length)
            # old算法（兼容模式）
            p.map_info_unchanged.add_weighted_edges_from([(sp, ep, length)])

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


def vehicle_load(p):
    # load from 'redis'
    pool = rds.ClusterConnectionPool(host=p.rds_connection, port=p.rds_port)
    connection = rds.RedisCluster(connection_pool=pool)
    v = connection.mget(keys=connection.keys(pattern=p.rds_search_pattern))#1000car
    vehicles = dict()
    for i in v:
        if not i:
            continue
        ii = json.loads(i)
        if ii.get('ohtID'):
            if ii['ohtStatus_OnlineControl'] != '1' or ii['ohtStatus_ErrSet'] != '0':
                continue
            vehicles[ii['ohtID']] = ii
            # existing vehicles in one path
            if not ii.get('mapId'):
                continue
            [sp, ep] = ii['mapId'].split('_')
            if not p.vehicle_jam.get((sp, ep)):
                p.vehicle_jam[sp, ep] = 1
            else:
                p.vehicle_jam[sp, ep] += 1
    log('Available vehicles number is', len(vehicles))
    p.vehicles = vehicles
    return p


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
    p.db_cursor.execute('SELECT * FROM OHTC_POSITION')
    df2 = pd.DataFrame(p.db_cursor.fetchall())
    station_location = dict()
    station_name = dict()
    for i in df2.index:
        num = df2[1][i]
        loc = df2[3][i]
        dft = df[(df[3] <= loc) & (df[4] >= loc)]
        station_location[num] = dft[1].values[0]
        station_name[num] = str(dft[3].values[0])
    # f1 = open('all_station.json', 'w')
    # f1.write(json.dumps(station_location))
    # f1.close()
    # f2 = open('stations_name.json', 'w')
    # f2.write(json.dumps(station_name))
    # f2.close()
    p.all_stations = station_location
    p.stations_name = station_name
    return p


def read_instructions(p):
    # oracle
    # p.db_cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS IN (0,1) and VEHICLE='0'") #筛选任务状态(0，1，2)
    p.db_cursor.execute("SELECT * FROM TRANSFER_TABLE WHERE STATUS=0 and VEHICLE='0'") #不筛选任务状态
    df = pd.DataFrame(p.db_cursor.fetchall())
    # debuger
    # if len(df) == 0:
    #     pass
    #     exit('All tasks have finished :)')
    # else:
    #     pass
    #     # print('Tasks number is', len(df))
    n = 0
    for i in df.index:
        tmp = p.Task()
        tmp.id = df[0][i]
        tmp.start_location = df[4][i]
        tmp.end_location = df[5][i]
        p.orders[tmp.id] = tmp
        n += 1
        # if n >= p.Control().task_num:
        #     break
    log.info(f'this is the task count:{n}')
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


