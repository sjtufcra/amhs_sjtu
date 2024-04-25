import heapq 
import networkx as nx
from typing import Dict, List, Tuple
import json
import hashlib
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


class AStart:
    def __init__(self):
        pass
    def manhattan_distance(self,node, goal_node):
        return abs(node.coordinates[0] - goal_node.coordinates[0]) + abs(node.coordinates[1] - goal_node.coordinates[1])
    def a_star_search(self,graph):
        open_set = []
        heapq.heappush(open_set,(0, graph.start_node))
        came_from = {graph.start_node.id: None}
        g_scores = {graph.start_node.id: 0}
        f_scores = {graph.start_node.id: (g_scores[graph.start_node.id]+graph.start_node.h)}

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)
            if current_node == graph.goal_node:
                path = []
                while current_node != graph.start_node:
                    # path.append(current_node)
                    path.append(current_node.id)
                    current_node = came_from[current_node.id]
                # path.append(graph.start_node)
                path.append(graph.start_node.id)
                path.reverse()
                return path

            for neighbor, edge_weight in graph.get_neighbors(current_node.id):
                tentative_g_score = g_scores[current_node.id] + edge_weight 

                if tentative_g_score < g_scores.get(neighbor.id, float('inf')):
                    came_from[neighbor.id] = current_node
                    g_scores[neighbor.id] = tentative_g_score
                    f_scores[neighbor.id] = tentative_g_score + neighbor.h
                    heapq.heappush(open_set, (f_scores[neighbor.id], neighbor))
        
        raise ValueError("No path found from start to goal.")

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
    
    def hash_index(self,input_string):
        hash_object = hashlib.sha256()
        hash_object.update(input_string.encode('utf-8'))
        hex_digest = hash_object.hexdigest()
        return hex_digest