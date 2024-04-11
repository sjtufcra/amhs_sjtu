from algorithm.A_start.graph.srccode import *


G = DiGraph()
G.add_nodes_from([7,6,2,1,0])
G.add_edges_from([(0,1),(0,2),(1,2),(1,3),(1,4),(2,3),(3,4)])
G.set_start_and_goal(G.nodes[0],G.nodes[4])

A = AStart()
path = A.a_star_search(G)
print(path)