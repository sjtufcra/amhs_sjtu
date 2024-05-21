import heapq 
import networkx as nx
from typing import Dict, List, Tuple
import json
import hashlib
import concurrent.futures
import itertools
from loguru import logger as log
class Node:
    def __init__(self, id: int,h_scores: int, coordinates: Tuple[float, float]):
        self.id = id
        self.h = h_scores
        self.g = float('inf')
        self.f = float('inf')
        self.coordinates = coordinates
    def __lt__(self, other):
        return self.f < other.f

class Edge:
    def __init__(self, start: Node, end: Node, weight: float):
        self.start = start
        self.end = end
        self.weight = weight


class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Node):
            return {
                '__type__': 'Node',
                'id': obj.id,
                'h_cost': str(obj.h) if obj.h == float("inf") else obj.h,
                'f_cost': str(obj.f) if obj.f == float("inf") else obj.f,
                'g_cost': str(obj.g) if obj.g == float("inf") else obj.g,
                'position': obj.coordinates
            }
        return super().default(obj)
class EdgeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Edge):
            return {
                '__type__': 'Edge',
                'start': obj.start.coordinates,
                'end': obj.end.coordinates,
                'weight': obj.weight
            }
        return super().default(obj)

class Graph():
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.adjacency_matrix: Dict[Tuple[int, int], float] = {}

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self.adjacency_matrix[(edge.start.id, edge.end.id)] = edge.weight
        # self.adjacency_matrix[(edge.end.id, edge.start.id)] = edge.weight  # 对于无向图，双向添加权重

    def get_neighbors(self, node_id: str):
        neighbors = []
        for other_id, weight in self.adjacency_matrix.items():
            if other_id[0] == node_id:
                neighbor_id = other_id[1]
                neighbor = next((n for n in self.nodes if n.id == neighbor_id), None)
                if neighbor is not None:
                    neighbors.append((neighbor, weight))
        return neighbors

    def set_start_and_goal(self, start_node: Node, goal_node: Node):
        self.start_node = start_node
        self.goal_node = goal_node
    def update_edge_weight(self, edge: Edge, new_weight: float):
        edge.weight = new_weight
    def update_adjacent_matrix(self, adjacent_matrix: dict):
        self.adjacent_matrix = adjacent_matrix

    def modify_adjacent_matrix_edge(self, edge: Edge, new_weight: float):
        self.adjacent_matrix[edge.start_node.id][edge.end_node.id] = new_weight
    
    def dfs_visit_nodes(self, start_node: Node) -> List[Node]:
        visited = set()
        stack = [start_node]

        nodes_list = []
        while stack:
            current_node = stack.pop()

            if current_node not in visited:
                visited.add(current_node)
                nodes_list.append(current_node)  # 添加当前节点到节点列表

                for neighbor, _ in self.get_neighbors(current_node.id):  # 修正：调用get_neighbors方法并解包邻居节点和权重
                    if neighbor not in visited:
                        stack.append(neighbor)  # 将邻居节点加入栈

        return nodes_list

# 
class NetworkXCompatibleGraph(Graph):
    def __init__(self):
        super().__init__()
    
    def to_networkx_graph(self):
        """将当前图转换为networkx.Graph对象"""
        G = nx.DiGraph() #有向图

        # 添加节点
        for node in self.nodes:
             G.add_node(node.id, coordinates=node.coordinates)

        # 添加边
        for edge in self.edges:
            G.add_edge(edge.start.id, edge.end.id, weight=edge.weight)

        return G

class DiGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.nodels: List[Node] = []
        self.edgels: List[Edge] = []
        self.adjacency_matrix: Dict[Tuple[str, str], float] = {}
   
    def add_nodes(self, node: Node):
        self.nodels.append(node)


    def add_node_from(self, start,end,length,cordinate=(0,0)):
        Start = Node(id=start,h_scores=1,coordinates=cordinate)
        End = Node(id=end,h_scores=1,coordinates=cordinate)
        if not self.has_node(start):
            self.add_nodes(Start)
        if not self.has_node(end):
            self.add_nodes(End)
        edg = Edge(start=Start,end=End,weight=length)
        self.add_edges(edg)

    def add_edges(self, edge: Edge):
        self.edgels.append(edge)
        self.adjacency_matrix[(edge.start.id, edge.end.id)] = edge.weight
        # self.adjacency_matrix[(edge.end.id, edge.start.id)] = edge.weight  # 对于无向图，双向添加权重

    def get_node_by_id(self, node_id: int):
        return next((n for n in self.nodels if n.id == node_id), None)

    def get_neighbors(self, node_id: str):
        neighbors = []
        for other_id, weight in self.adjacency_matrix.items():
            if other_id[0] == node_id:
                neighbor_id = other_id[1]
                neighbor = next((n for n in self.nodels if n.id == neighbor_id), None)
                if neighbor is not None:
                    neighbors.append((neighbor, weight))
        return neighbors

    def set_start_and_goal(self, start_node: Node, goal_node: Node):
        self.start_node = start_node
        self.goal_node = goal_node
    def update_edge_weight(self, edge: Edge, new_weight: float):
        edge.weight = new_weight
    def update_adjacent_matrix(self, adjacent_matrix: dict):
        self.adjacent_matrix = adjacent_matrix

    def modify_adjacent_matrix_edge(self, edge: Edge, new_weight: float):
        self.adjacent_matrix[edge.start_node.id][edge.end_node.id] = new_weight
    # dijkstra算法
    def dijkstra(self, start_node: Node, end_node: Node):
        # 初始化距离字典，源点到自身的距离为0，其余为无穷大
        distances = {node.id: float('infinity') for node in self.nodels}
        distances[start_node.id] = 0
        came_from = {}
        # 初始化优先队列和已访问集合
        pq = [(0, start_node)]
        visited = set()

        while pq:
            # 获取当前最短距离的节点
            current_distance, current_node = heapq.heappop(pq)

            # 如果节点已经被访问过，跳过
            if current_node in visited:
                continue

            # 将当前节点标记为已访问
            visited.add(current_node)

            # 更新与当前节点相邻节点的距离
            for neighbor, weight in self.get_neighbors(current_node.id):
                distance = current_distance + weight
                # 如果找到了更短的路径，则更新距离并将其加入优先队列
                if distance < distances[neighbor.id]:
                    distances[neighbor.id] = distance
                    heapq.heappush(pq, (distance, neighbor))
                    # 记录路径来源
                    came_from[neighbor.id] = current_node

        # 回溯路径
        path = []
        current = end_node.id
        while current != start_node.id:
            path.append(current)
            current = came_from[current].id

        path.append(start_node.id)
        path.reverse()

        return path
    
    # Floyd-Warshall算法
    def floyd_warshall(self):
        distances = {node: {other: float('infinity') for other in self.nodes} for node in self.nodes}
        for node in self.nodes:
            distances[node][node] = 0

        # 动态规划循环
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    # 如果通过中间节点k的路径更短，则更新最短路径
                    distances[i][j] = min(
                        distances[i][j],
                        distances[i][k] + distances[k][j]
                    )

        return distances

    # DP 寻找最短路径
    def get_shortest_path_between(self, start_id, end_id):
        all_distances = self.floyd_warshall()
        path = [end_id]
        current_node = end_id

        while current_node != start_id:
            for neighbor, distance in all_distances[start_id].items():
                if distance == all_distances[start_id][current_node] - all_distances[current_node][end_id]:
                    path.append(neighbor)
                    current_node = neighbor
                    break

        path.reverse()  # 使路径从源到目标
        return path

class AStart:
    def __init__(self):
        self.cache = {}
        pass
    def manhattan_distance(self,node, goal_node):
        return abs(node.coordinates[0] - goal_node.coordinates[0]) + abs(node.coordinates[1] - goal_node.coordinates[1])
# no-cache
    def a_star_search(self,graph,bidirectional=True):
        if bidirectional:
            return self.a_star_search_cache(graph)
        else:
            return self.a_star_search_nocache(graph)
# no-cache
    def a_star_search_nocache(self,graph):
        open_set = []
        heapq.heappush(open_set,(0, graph.start_node))
        came_from = {graph.start_node.id: None}
        g_scores = {graph.start_node.id: 0}
        f_scores = {graph.start_node.id: (g_scores[graph.start_node.id]+graph.start_node.h)}
        visited = set()
        
        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)
            if current_node == graph.goal_node:
                path = self._reconstruct_path(came_from,graph.start_node, graph.goal_node)
                return path

            # for neighbor, edge_weight in graph.get_neighbors(current_node.id):
            #     tentative_g_score = g_scores[current_node.id] + edge_weight 

            #     if tentative_g_score < g_scores.get(neighbor.id, float('inf')):
            #         came_from[neighbor.id] = current_node
            #         g_scores[neighbor.id] = tentative_g_score
            #         f_scores[neighbor.id] = tentative_g_score + neighbor.h
            #         heapq.heappush(open_set, (f_scores[neighbor.id], neighbor))
            if current_node.id not in visited:
                visited.add(current_node.id)
                futures = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for neighbor, edge_weight in graph.get_neighbors(current_node.id):
                        futures.append(executor.submit(self.update_scores, current_node, neighbor, edge_weight, g_scores, f_scores, came_from))

                for future in concurrent.futures.as_completed(futures):
                    neighbor_id, _, _ = future.result()
                    heapq.heappush(open_set, (f_scores[neighbor_id], graph.get_node_by_id(neighbor_id)))
        log.error("No path found from start to goal.")
# cache
    def a_star_search_cache(self, graph):
            open_set_start = [(0, graph.start_node)]
            open_set_goal = [(0, graph.goal_node)]
            came_from_start = {graph.start_node.id: None}
            came_from_goal = {graph.goal_node.id: None}
            g_scores_start = {graph.start_node.id: 0}
            g_scores_goal = {graph.goal_node.id: 0}
            f_scores_start = {graph.start_node.id: (g_scores_start[graph.start_node.id] + graph.start_node.h)}
            f_scores_goal = {graph.goal_node.id: (g_scores_goal[graph.goal_node.id] + graph.goal_node.h)}
            visited_start = set()
            visited_goal = set()

            while open_set_start and open_set_goal:
                # Search from start node
                current_f_score, current_node = heapq.heappop(open_set_start)
                if current_node.id not in visited_start:
                    visited_start.add(current_node.id)
                    if current_node == graph.goal_node:
                        path = [graph.start_node.id]
                        while current_node != graph.start_node:
                            path.append(current_node.id)
                            current_node = came_from_start[current_node.id]
                        path.reverse()
                        return path

                    for neighbor, edge_weight in graph.get_neighbors(current_node.id):
                        self.update_scores_cache(current_node, neighbor, edge_weight, g_scores_start, f_scores_start, came_from_start)

                # Search from goal node
                current_f_score, current_node = heapq.heappop(open_set_goal)
                if current_node.id not in visited_goal:
                    visited_goal.add(current_node.id)
                    if current_node == graph.start_node:
                        path = [graph.goal_node.id]
                        while current_node != graph.goal_node:
                            path.append(current_node.id)
                            current_node = came_from_goal[current_node.id]
                        path.reverse()
                        return path

                    for neighbor, edge_weight in graph.get_neighbors(current_node.id):
                        self.update_scores_cache(current_node, neighbor, edge_weight, g_scores_goal, f_scores_goal, came_from_goal)

            log.error("No path found from start to goal.")
    def _generate_cache_key(self, current_node_id, neighbor_id):
        """生成缓存键的辅助函数，使用SHA-256提高安全性"""
        data = (current_node_id, neighbor_id)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

# cache_scores
    def update_scores_cache(self, current_node, neighbor, edge_weight, g_scores, f_scores, came_from):
        if not hasattr(neighbor, 'id') or current_node.id is None:
            log.error("neighbor and current_node must have 'id' attribute")

        neighbor_id = neighbor.id
        cache_key = self._generate_cache_key(current_node.id, neighbor_id)

        if cache_key in self.cache:
            tentative_g_score, path = self.cache[cache_key]
        else:
            tentative_g_score = g_scores.get(current_node.id, 0) + edge_weight
            path = [current_node.id] if current_node.id != neighbor_id else []
            self.cache[cache_key] = (tentative_g_score, path)

        # 检查是否发现更短的路径
        if tentative_g_score < g_scores.get(neighbor_id, float('inf')):
            if neighbor_id not in came_from:
                came_from[neighbor_id] = current_node
            g_scores[neighbor_id] = tentative_g_score
            f_scores[neighbor_id] = tentative_g_score + self.cache[cache_key][1]

            # 更新完整路径
            path.extend([current_node.id])
            self.cache[cache_key] = (tentative_g_score, path)

        return neighbor_id, g_scores[neighbor_id], f_scores[neighbor_id]
    # 溯源Path
    def _reconstruct_path(self, came_from, start_node, goal_node):
        """重构路径函数，优化可读性和代码复用"""
        current_node = goal_node
        path = [current_node.id]
        while current_node != start_node:
            current_node = came_from[current_node.id]
            path.append(current_node.id)
        path.reverse()
        return path
    # 更新数据
    def update_scores(self, current_node, neighbor, edge_weight, g_scores, f_scores, came_from):
        tentative_g_score = g_scores[current_node.id] + edge_weight
        neighbor_id = neighbor.id
        if tentative_g_score < g_scores.get(neighbor_id, float('inf')):
            came_from[neighbor_id] = current_node
            g_scores[neighbor_id] = tentative_g_score
            f_scores[neighbor_id] = tentative_g_score + neighbor.h
        return neighbor_id, g_scores[neighbor_id], f_scores[neighbor_id]
    def sum_node_path(self,graph):
        indexset = set()
        pathset = set()
        jsondata = dict(OHTC_PATH=[])
        startls = graph.nodels.copy()
        for k,node in enumerate(startls):
            if node.id[0]!='W':
                kin = startls.index(node)
                startls.pop(kin)
                continue
            start = node
            endnodes = startls.copy()
            sin = endnodes.index(start)
            endnodes.pop(sin)
            for end in endnodes:
                if end.id[0]!='W':
                    ein = endnodes.index(end)
                    endnodes.pop(ein)
                    continue
                if start != end:
                    graph.set_start_and_goal(start,end)
                    path = self.a_star_search(graph)
                    pathdata = ','.join(path)
                    pathindex = ','.join([start.id,end.id])
                    index = self.hash_index(pathindex)
                    indexset.add(index)
                    pathset.add(pathdata)
                    jsondata['OHTC_PATH'].append({f'{index}': pathdata})
        
        with open(f'./outpath.json', 'w') as f:
            json.dump(jsondata, f)
        return indexset,pathset
    
    def more_path(self,graph):
        newpathNode = []
        for x in graph.nodels:
            if x.id[0]=='W':
                newpathNode.append(x)
        startls = newpathNode.copy()

         # 创建线程池
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用itertools.combinations生成所有可能的元素对
            element_pairs = itertools.combinations(startls, 2)
            
            # 将计算任务提交给线程池
            executor.map(lambda pair: self.computePath(pair,graph), element_pairs)
        # 打印结果或根据需要处理结果
        log.info("All tasks completed.")

    def hash_index(self,input_string):
        hash_object = hashlib.sha256()
        hash_object.update(input_string.encode('utf-8'))
        hex_digest = hash_object.hexdigest()
        return hex_digest
    
    def computePath(self,data,graph):
        start,end = data
        if start != end:
            graph.set_start_and_goal(start,end)
            path = self.a_star_search(graph)
            pathdata = ','.join(path)
            pathindex = ','.join([start.id,end.id])
            index = self.hash_index(pathindex)
            with open(f'./outpath.txt', 'a') as f:
                f.write("{"+f'{index}: {pathdata}'+'}'+'\n')
