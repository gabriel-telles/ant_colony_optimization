from AntColonyOptimization import AntColonyOptimization

graph = {
    0: [1, 9],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 6],
    7: [6, 8],
    8: [7, 13],
    9: [0, 10, 14],
    10: [9, 11, 12, 17],
    11: [10, 12],
    12: [10, 11, 13, 18],
    13: [8, 12, 16],
    14: [9, 15, 16, 17],
    15: [14, 16],
    16: [13, 14, 15, 18],
    17: [10, 14, 18],
    18: [12, 16, 17],
}

aco = AntColonyOptimization(graph, source=0, destination=8)
aco.run()
print(aco.pheromone)