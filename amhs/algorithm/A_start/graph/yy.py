def floyd_warshall(graph):
    """
    graph: 一个二维列表表示的邻接矩阵，其中graph[i][j]表示顶点i到顶点j的边的权重。
           如果两点之间没有直接连接，则值为正无穷大（通常用float('inf')表示）。
    返回: dist，一个二维列表，dist[i][j]表示顶点i到顶点j的最短路径长度。
    """
    # 初始化距离矩阵
    dist = graph.copy()
    
    # 图的顶点数
    num_vertices = len(graph)
    
    # 自身到自身的距离为0
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j:
                dist[i][j] = 0
                
    # 通过中间顶点k来更新最短路径
    for k in range(num_vertices):
        # 选择所有顶点对(i, j)
        for i in range(num_vertices):
            for j in range(num_vertices):
                # 如果经过顶点k的路径更短，则更新dist[i][j]
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                
    return dist

# 示例图的邻接矩阵表示
# 假设我们有一个图，顶点编号为0, 1, 2
#          0  1  2
#       -------------
#    0 | 0  5  inf
#    1 | 5  0  3
#    2 | inf 3  0
example_graph = [
    [0, 5, float('inf')],
    [5, 0, 3],
    [float('inf'), 3, 0]
]

# 使用Floyd-Warshall算法计算最短路径
shortest_paths = floyd_warshall(example_graph)

# 打印结果
for row in shortest_paths:
    print(row)