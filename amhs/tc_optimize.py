import networkx as nx
import math
import time
import random
import concurrent.futures
import multiprocessing
from loguru import logger as log

from tc_out import *
from tc_in import *
from algorithm.A_start.graph.srccode import *


def task_assign(p, use_multiprocessing=True):
    g = 0
    while p.runBool:
        start_time = time.time()
        p.map_info = p.map_info_unchanged
        p = vehicle_load(p)
        # refresh tasks
        p = read_instructions(p)
        # revise map info
        p.used_vehicle = set()
        # revise map info
        j, n = 0, 0
        car = 0
        log.info(f"algorithm:{p.algorithm_on},task:{len(p.orders)}")
        log.info(f"algorithm,task_time:{time.time() - start_time}")
        # 
        if use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                try:
                    future_to_order_id = {
                        executor.submit(process_order, v, p): order_id
                        for order_id, v in p.orders.items() if v.finished == 0
                    }

                # results = []
                # for future in concurrent.futures.as_completed(future_to_order_id):
                #     order_id = future_to_order_id[future]
                #     try:
                #         finished = future.result()
                #         if finished:
                #             n += 1
                #         results.append((order_id, finished))
                except Exception as exc:
                    log.error(f"Order processing generated an exception: {exc}")

                # for order_id, finished in results:
                #     if finished:
                #         v = p.orders[order_id]
                #         v.finished = 1
                #         car += 1
        else:
            for k, v in p.orders.items():
                if v.finished == 0:
                    # veh, v0 = vehicle_select(v, p)  # getpath
                    veh, v0 = vehicle_select_fast_random(v, p)  # getpath
                    start, end = terminus_select(j, v0, p, v)
                    v.vehicle_assigned = veh
                    v.delivery_route = shortest_path(start, end, p, v, typ=0)
                    if p.mode == False:
                        log.info(f"alltime,task_time:{time.time() - start_time}")
                        log.info(f'success:{k},{v}')
                        # output_new(p, k, v)
                    else:
                        output_new(p, k, v)
                        pass
                    # v.finished = 1
                    car += 1
        log.info(f"model:{p.algorithm_on},task_time:{time.time() - start_time}")


# new_function_static
def epoch_static(p):
    while p.runBool:
        log.info(f"开始运行算法")
        start_time = time.time()
        p.map_info = p.map_info_unchanged
        # load less than 10 tasks
        t0 = time.time()
        p = read_instructions_static(p)
        log.info(f"本轮读取任务时长:{time.time() - t0}")
        # added in 20240607, only select stations used instead of all
        if p.debug_on:
            p = track_generate_station_new(p)
        t1 = time.time()
        if len(p.taskList) > 0:
            vehicle_load_static(p)
        log.info(f"本轮分配任务时长:{time.time() - t1}")
        log.info(f"本轮算法执行时长:{time.time() - start_time}")
    return 0


def track_generate_station_new(p):
    tasks = p.taskList
    station_location = dict()
    station_name = dict()
    if len(tasks) > 0:
        df = p.original_map_info
        with p.db_pool.get_connection() as db_conn:
            cursor = db_conn.cursor()
            for task in tasks:
                s = [task.start_location, task.end_location]
                for s0 in s:
                    sql = f"SELECT * FROM OHTC_POSITION WHERE MCSPOINT IN (\'{s0}\')"
                    cursor.execute(sql)
                    df2 = pd.DataFrame(cursor.fetchall())
                    for i in df2.index:
                        num = df2[1][i]
                        loc = df2[3][i]
                        dft = df[(df[3] <= loc) & (df[4] >= loc)]
                        station_location[num] = dft[1].values[0]
                        station_name[num] = str(dft[3].values[0])
            db_conn.commit()
            cursor.close()
    p.all_stations = station_location
    p.stations_name = station_name
    return p


def process_order(v, p):
    start_time = time.time()
    if v.finished == 0:
        veh, v0 = vehicle_select_fast_random(v, p)  # getpath
        start, end = terminus_select(0, v0, p, v)
        v.vehicle_assigned = veh
        v.delivery_route = shortest_path(start, end, p, v, typ=0)
        if p.mode == False:
            log.info(f'success:{v.id},{v}')
            log.info(f"model:{p.algorithm_on},task_time:{time.time() - start_time}")
        else:
            output_new(p, v.id, v)
            pass
    return


# 版本回退用：多线程
def process_order_static(v, p):
    start_time = time.time()
    if v.finished == 0:
        veh, v0 = vehicle_select_fast_random(v, p)  # getpath
        start, end = terminus_select(0, v0, p, v)
        v.vehicle_assigned = veh
        v.delivery_route = shortest_path(start, end, p, v, typ=0)
        if p.mode == False:
            log.info(f'success:{v.id},{v}')
            log.info(f"model:{p.algorithm_on},task_time:{time.time() - start_time}")
        else:
            output_new(p, v.id, v)
            pass
    return


# random select
def vehicle_select_fast_random(task, p):
    vs0 = get_vehicles_from_bay_fast(task.task_bay, p)
    veh = None
    if isinstance(vs0, list):
        value = random.choice(vs0)
        if isinstance(value, list) or isinstance(value, tuple):
            veh = value[11]
        else:
            veh = value.get("ohtID")
    else:
        keys_to_choose_from = list(vs0.keys())
        veh = random.choice(keys_to_choose_from)
    p.used_vehicle.add(veh)
    return veh, p.vehicles_get[veh]


def terminus_select(j, v0, p, v):
    if j == 0:
        # routh: position -> start
        if p.mode == 1:
            postion = v0.get('position')
        else:
            postion = v0[43]
        start = vehicle_node_search(postion, p)
        end = p.all_stations[v.start_location]
    else:
        # routh: start -> end
        start = p.all_stations[v.start_location]
        end = p.all_stations[v.end_location]
    return start, end


def shortest_path(start, end, p, v, typ=0):
    st = time.time()
    if typ == 0:
        # only return the path
        if p.algorithm_on is not None:
            path = algorithm_on(p, start, end)
            path.append(p.stations_name[v.end_location])
            log.info(f'sho,time:{time.time() - st}')
            return path
    else:
        # return the length
        if p.algorithm_on is not None:
            path0 = algorithm_on(p, start, end)
        path = 0
        for i in range(len(path0) - 1):
            idx = path0[i:i + 2]
            try:
                path += (p.map_info.edges[idx]).get('weight')
            except:
                path += 0
        return path


def vehicle_node_search(position, p):
    position = float(position)
    df = p.original_map_info
    # NT_point
    node = df[(df[3] <= position) & (df[4] >= position)][2].values[0]
    return node


def get_new_bay(bay, p, used_bays):
    new_bay = None
    tmp = p.bays_relation.get(bay)
    for i in tmp:
        if i not in used_bays:
            new_bay = bay
    if not new_bay:
        for bay in tmp:
            new_bay = get_new_bay(bay, p, used_bays)
    return new_bay


def get_vehicles_from_bay(bay, p):
    vs0 = dict()
    used_bays = set()
    for k0, v0 in p.vehicles_get.items():
        if p.mode == 1:
            tmp = v0.get('bay')
            if not tmp:
                continue
            if tmp in [bay] and (k0 not in p.used_vehicle):
                vs0[k0] = v0
        else:
            if not v0[1]:
                continue
            if v0[1] in [bay] and (k0 not in p.used_vehicle):
                vs0[k0] = v0
    if len(vs0) <= 0:
        vs0 = p.vehicles_get
    return vs0


# fast get vehicles
def get_vehicles_from_bay_fast(bay, p):
    if bay is None:
        return p.vehicles_get
    else:
        vehicle_list = p.vehicles_bay_get.get(bay);
        if vehicle_list is None:
            return p.vehicles_get
        return vehicle_list


def algorithm_on(p, start, end):
    if p.algorithm_on == 2:
        # A*算法
        st = time.time()
        p.map_info.set_start_and_goal(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
        path = p.Astart.a_star_search(p.map_info)
        log.info(f'A*算法,time:{time.time() - st}')
    elif p.algorithm_on == 3:
        # dijkstra算法
        path = p.map_info.dijkstra(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
    elif p.algorithm_on == 4:
        # DP 算法
        path = p.map_info.get_shortest_path_between(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
    else:
        p.algorithm_on == 1
        path = nx.shortest_path(p.map_info, source=start, target=end)
    return path
