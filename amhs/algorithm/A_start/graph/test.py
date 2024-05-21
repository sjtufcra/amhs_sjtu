import concurrent.futures
import itertools
import heapq

def calculate_sum(pair,no):
    """计算一对元素的和"""
    x, y = pair
    return x + y

def process_data(data):
    """
    使用多线程遍历数据列表，计算每对元素的和。
    
    :param data: 长度为2000的列表
    """
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用itertools.combinations生成所有可能的元素对
        element_pairs = itertools.combinations(data, 2)
        
        # 将计算任务提交给线程池
        results = list(executor.map(calculate_sum, element_pairs))
        
    # 打印结果或根据需要处理结果
    for result in results:
        print(result)

def dijkstra(graph, start):
    # 初始化距离字典，源点到自身的距离为0，其余为无穷大
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    # 初始化优先队列和已访问集合
    pq = [(0, start)]
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
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # 如果找到了更短的路径，则更新距离并将其加入优先队列
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
# # 示例数据
# data_example = [i for i in range(10)]

# # 调用函数
# process_data(data_example)

# 示例图，以字典形式表示，键为节点，值为另一个字典，表示邻居节点及其距离
graph_example = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 1},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 1, 'C': 1}
}

# 使用Dijkstra算法计算从节点'A'到其他节点的最短距离
shortest_distances = dijkstra(graph_example, 'A')
print(shortest_distances)

class Graph:
    # ...其他代码不变...

    def dijkstra(self, start_node: Node, end_node: Node):
        # 初始化距离字典，源点到自身的距离为0，其余为无穷大
        distances = {node: float('infinity') for node in self.nodes}
        distances[start_node] = 0

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
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
                    # 记录路径来源
                    came_from[neighbor] = current_node

        # 回溯路径
        path = []
        current = end_node
        while current != start_node:
            path.append(current)
            current = came_from[current]

        path.append(start_node)
        path.reverse()

        return path
    
    def floyd_warshall(self):
        # 初始化邻接矩阵
        adjacency_matrix = self.adjacency_matrix.copy()

        # 获取图中的节点数量
        num_nodes = len(adjacency_matrix)

        # 动态规划循环
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # 如果通过中间节点k的路径更短，则更新最短路径
                    adjacency_matrix[i][j] = min(
                        adjacency_matrix[i][j],
                        adjacency_matrix[i][k] + adjacency_matrix[k][j]
                    )

        return adjacency_matrix

    def get_shortest_path_between(self, start_id, end_id):
        adjacency_matrix = self.floyd_warshall()

        # 从邻接矩阵中获取最短路径
        shortest_path = adjacency_matrix[start_id][end_id]

        return shortest_path

# 示例：
graph = Graph()
# 添加节点和边...
start_node = next((n for n in graph.nodes if n.id == 'A'), None)
end_node = next((n for n in graph.nodes if n.id == 'D'), None)
shortest_path = graph.dijkstra(start_node, end_node)
print(shortest_path)