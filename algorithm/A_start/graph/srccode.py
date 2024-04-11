import heapq 
import networkx as nx
from typing import Dict, List, Tuple
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
import json

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
        # 如果遇到其他未知类型，调用基类方法
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
        # 如果遇到其他未知类型，调用基类方法
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
        # self.adjacency_matrix[(edge.end.id, edge.start.id)] = edge.weight  # 对于无向图，双向添加权重

    def get_neighbors(self, node_id: int) -> List[Tuple[Node, float]]:
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
    def get_neighbors(self, node_id: int) -> List[Tuple[Node, float]]:
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


class AStar:
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
                    path.append(current_node)
                    current_node = came_from[current_node.id]
                path.append(graph.start_node)
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