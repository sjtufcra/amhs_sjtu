import heapq

class Node:
    def __init__(self, x, y):
            self.x = x
            self.y = y
            self.g = float('inf')  
            self.h = float('inf')  
            self.f = float('inf')  
            self.parent = None  # 初始化父节点为None，用于记录路径
    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def a_star_search(start, goal, grid, obstacles=None):
    if obstacles is None:
        obstacles = set()
    open_list = []
    heapq.heappush(open_list, (start.h, start))

    closed_list = {}  # 使用字典存储节点及其对应的g值，便于快速查找和更新

    start.g = 0
    start.h = heuristic(start, goal)
    start.f = start.g + start.h
    grid[start.x][start.y]=start
    grid[goal.x][goal.y]=goal

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current is not None:
                path.append((current.x, current.y))
                current = current.parent
            path.reverse()
            return path

        closed_list[current] = current.g  # 将节点及其g值添加到closed_list

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor_x = current.x + dx
            neighbor_y = current.y + dy

            if neighbor_x < 0 or neighbor_y < 0 or \
                    neighbor_x >= len(grid) or neighbor_y >= len(grid[0]) or \
                    (neighbor_x, neighbor_y) in obstacles:
                continue
            if current.parent and (neighbor_x, neighbor_y) == (current.parent.x,current.parent.y):
                    continue

            neighbor = grid[neighbor_x][neighbor_y]

            tentative_g = current.g + 1  # 假设移动成本为1

            if neighbor in closed_list and tentative_g >= neighbor.g:  # 更改此处的判断逻辑
                continue

            neighbor.g = tentative_g
            neighbor.h = heuristic(neighbor, goal)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current

            if neighbor not in open_list:
                heapq.heappush(open_list, (neighbor.f, neighbor))

    return None