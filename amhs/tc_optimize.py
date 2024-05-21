import networkx as nx
import math
import time
import concurrent.futures
import multiprocessing
from loguru import logger as log

from tc_out import *
from tc_in import *
from algorithm.A_start.graph.srccode import *

def task_assign(p, use_multiprocessing=True):
        start_time = time.time()
        g = 0
        while p.runBool:
            p.map_info = p.map_info_unchanged
            p = vehicle_load(p)
            # refresh tasks
            p = read_instructions(p)
            # revise map info
            p.used_vehicle = set()
            # revise map info
            j, n = 0, 0
            car = 0
        # 
            if use_multiprocessing:
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_to_order_id = {
                        executor.submit(process_order, order_id, p, car): order_id
                        for order_id, v in p.orders.items() if v.finished == 0
                    }

                    results = []
                    for future in concurrent.futures.as_completed(future_to_order_id):
                        order_id = future_to_order_id[future]
                        try:
                            finished = future.result()
                            if finished:
                                n += 1
                            results.append((order_id, finished))
                        except Exception as exc:
                            log.error(f"Order {order_id} processing generated an exception: {exc}")

                    for order_id, finished in results:
                        if finished:
                            v = p.orders[order_id]
                            v.finished = 1
                            car += 1
            else:
                for k, v in p.orders.items():
                    if v.finished == 0:
                        veh, v0 = vehicle_select(v, p)  # getpath
                        start, end = terminus_select(j, v0, p, v)
                        v.vehicle_assigned = veh
                        v.delivery_route = shortest_path(start, end, p, v, typ=0)
                        if p.mode == 1:
                            log.info(f'success:{k},{v}')
                        else:
                            output_new(p, k, v)
                        v.finished = 1
                        car += 1
            g+=n
            if g == 20000:
                log.info(f'endtime:{time.time()-start_time},task_sum:{g}')
                p.runBool = False

def process_order(order_id, p, car):
    v = p.orders[order_id]

    if v.finished == 0:
        # old
        # veh, v0 = vehicle_select(v, p)  # getpath
        # new_fast
        veh, v0 = vehicle_select_fast(v, p)  # getpath
        start, end = terminus_select(0, v0, p, v)
        v.vehicle_assigned = veh
        v.delivery_route = shortest_path(start, end, p, v, typ=0)
        if p.mode == 1:
            log.info(f'success:{order_id},{v}')
        else:
            output_new(p, order_id, v)
        v.finished = 1
        car += 1

    return v.finished

def vehicle_select(task, p):
    # task_bay = task.start_location.split('_')[1]
    # old function
    vs0 = get_vehicles_from_bay(task.task_bay, p)
    veh = None
    veh_len = math.inf
    for k, v in vs0.items():
        start, end = terminus_select(0, v, p, task)
        length = shortest_path(start, end, p, task, typ=1)
        if length < veh_len:
            veh_len = length
            veh = k
    p.used_vehicle.add(veh)
    return veh, p.vehicles_get[veh]
# fast seclect
def vehicle_select_fast(task, p):
    # task_bay = task.start_location.split('_')[1]
    # old function
    # vs0 = get_vehicles_from_bay(task.task_bay, p)
    # fix: new function
    vs0 = get_vehicles_from_bay_fast(task.task_bay, p)
    veh = None
    veh_len = math.inf
    if isinstance(vs0,list):
        for value in vs0:
            start, end = terminus_select(0, value, p, task)
            length = shortest_path(start, end, p, task, typ=1)
            if length < veh_len:
                veh_len = length
                if isinstance(value,list):
                    veh = value[11]
                else:
                    veh = value["ohtID"]
    else:
        for k, v in vs0.items():
            start, end = terminus_select(0, v, p, task)
            length = shortest_path(start, end, p, task, typ=1)
            if length < veh_len:
                veh_len = length
                veh = k
    p.used_vehicle.add(veh)
    return veh, p.vehicles_get[veh]

def terminus_select(j, v0, p, v):
    if j == 0:
        # routh: position -> start
        if p.mode == 1:
            postion = v0['position']
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
    if typ == 0:
        # only return the path
        if p.algorithm_on is not None:
            path = algorithm_on(p,start,end)
            path.append(p.stations_name[v.end_location])
            return path
    else:
        # return the length
        if p.algorithm_on is not None:
            path0 = algorithm_on(p,start,end)
        path = 0
        for i in range(len(path0) - 1):
            idx = path0[i:i + 2]
            path += p.map_info.edges[idx]['weight']
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
            if not v0.get('bay'):
                continue
            if v0['bay'] in [bay] and (k0 not in p.used_vehicle):
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
    
def algorithm_on(p,start,end):
    if p.algorithm_on == 2:
        # A*算法
        p.map_info.set_start_and_goal(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
        path = p.Astart.a_star_search(p.map_info)
    elif p.algorithm_on == 3:
        # dijkstra算法
        path = p.map_info.dijkstra(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
    elif p.algorithm_on == 4:
        # DP 算法
        path = p.map_info.get_shortest_path_between(p.map_info.get_node_by_id(start), p.map_info.get_node_by_id(end))
    else:
        path = nx.shortest_path(p.map_info, source=start, target=end)
    return path