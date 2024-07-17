import random

num_nodes = 30
num_edges = 200
max_cost = 100
adjacency_list = {i: [] for i in range(num_nodes)}
edge_cost_dict = {}

possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
random.shuffle(possible_edges)
edges = possible_edges[:num_edges]

for (u, v) in edges:
    cost = random.randint(1, max_cost)
    adjacency_list[u].append((v, cost))
    adjacency_list[v].append((u, cost))
    edge_cost_dict[(u, v)] = cost
    edge_cost_dict[(v, u)] = cost
