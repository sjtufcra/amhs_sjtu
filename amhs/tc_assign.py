import networkx as nx
from loguru import logger as log
import math

from .tc_out import *
from .tc_in import *
from .algorithm.A_start.graph.srccode import *


# def task_assign(p):
#     p.map_info = p.map_info_unchanged
#     # revise map info
#     # for k, v in p.vehicle_jam.items():
#     #     w0 = p.map_info.edges[k[0], k[1]]['weight'] * (1 + v)
#     #     p.map_info.add_weighted_edges_from([(k[0], k[1], w0)])
#     j, n = 0, 0
#     for k, v in p.orders.items():
#         # specific position of task
#         if v.finished == 0:
#             veh, v0 = vehicle_select(v, p)
#             start, end = terminus_select(j, v0, p, v)
#             v.vehicle_assigned = veh
#             v.delivery_route = shortest_path(start, end, p, v)
#             output_new(p, k, v)
#             v.finished = 1
#         if v.finished == 1:
#             n += 1
#     return p


def task_assign_new(p):
    g, max_g = 1, 6
    # t = time.process_time()
    while p.runBool:
        log.info(f'run status: {p.runBool}')
        # refresh vehicles
        p = vehicle_load(p)
        # refresh tasks
        p = read_instructions(p)
        # revise map info
        
        p.map_info = revise_map_info(p) #更新边的权重值，方便后续路径规划
        # refresh before assigning
        p.used_vehicle = set()
        j, n = 0, 0
        # this is the car count
        car = 0
        # refresh orders
        for k, v in p.orders.items():
            if v.finished == 0:
                veh, v0 = vehicle_select(v, p)
                start, end = terminus_select(j, v0, p, v)
                v.vehicle_assigned = veh
                v.delivery_route = shortest_path(start, end, p, v)
                output_new(p, k, v)#正常作业流程
                v.finished = 1
                car +=1
            if v.finished == 1:
                n += 1
        g += 1
        log.info(f'this is the car count in the mission batch and the task count: car[{car}],batch[{g}],task_sum[{n}]')
    return p


def revise_map_info(p):
    # p.map_info = copy.deepcopy(p.map_info_unchanged)
    # for k, v in p.vehicle_jam.items():
    #     w0 = p.map_info.edges[k[0], k[1]]['weight'] * (1 + v)
    #     p.map_info.add_weighted_edges_from([(k[0], k[1], w0)])
    map_info = p.map_info_unchanged
    return map_info


def vehicle_select(task, p):
    task_bay = task.start_location.split('_')[1]
    vs0 = get_vehicles_from_bay(task_bay, p)
    veh = None
    veh_len = math.inf
    for k, v in vs0.items():
        start, end = terminus_select(0, v, p, task)
        length = shortest_path(start, end, p, task, typ=1)
        if length < veh_len:
            veh_len = length
            veh = k
    p.used_vehicle.add(veh)
    return veh, p.vehicles[veh]


def terminus_select(j, v0, p, v):
    if j == 0:
        # routh: position -> start
        start = vehicle_node_search(v0['position'], p)
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
            if p.algorithm_on == 2:
                # A*算法
                path = nx.shortest_path(p.map_info, source=start, target=end)
                pass
            else:
                try:
                    path = nx.shortest_path(p.map_info, source=start, target=end)
                    path.append(p.stations_name[v.end_location])
                    return path
                except Exception as e:
                    log.error(f'error{e}')
                    # print(e, start, end)
        # path_check(path, p, v.id)
        # path = nx.dijkstra_path(p.map_info, source=start, target=end)
        # add station ID at the end of path list
        # path.append(p.stations_name[v.end_location])
    else:
        # return the length
        if p.algorithm_on is not None:
            if p.algorithm_on == 2:
                # A*算法
                path0 = nx.shortest_path(p.map_info, source=start, target=end)
                pass
            else:
                path0 = nx.shortest_path(p.map_info, source=start, target=end)
        # path = nx.dijkstra_path_length(p.map_info, source=start, target=end)
        path = 0
        for i in range(len(path0)-1):
            idx = path0[i:i + 2]
            path += p.map_info.edges[idx]['weight']
        # path = sum(p.map_info.edges[path0[i:i + 2]]['weight'] for i in range(len(path0)-1))
        return path


def path_check(path, p, task_id):
    for i in range(len(path) - 1):
        idx = path[i:i + 2]
        if not p.map_info.edges.get(idx):
            pass
            # print('Mission {} path error'.format(task_id))
    return 0


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
    # searching available vehicles
    vs0 = dict()
    used_bays = set()
    # searching bays
    # while 1:
    #     for k0, v0 in p.vehicles.items():
    #         if v0['bay'] in [bay] and (k0 not in p.used_vehicle):
    #             vs0[k0] = v0
    #     if len(vs0) <= 0:
    #         bay = get_new_bay(bay, p, used_bays)
    #     else:
    #         break
    #
    # searching all veh
    for k0, v0 in p.vehicles.items():
        if not v0.get('bay'):
            continue
        if v0['bay'] in [bay] and (k0 not in p.used_vehicle):
            vs0[k0] = v0
    if len(vs0) <= 0:
        vs0 = p.vehicles
    return vs0
