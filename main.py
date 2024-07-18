import matplotlib.pyplot as plt
import numpy as np
from AntColonyOptimization import AntColonyOptimization
from collections import defaultdict

def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def run_simulations(graph, cost, source, destination, num_iterations, parameter_name, parameter_values):
    results = defaultdict(list)
    for value in parameter_values:
        if parameter_name == 'alpha':
            aco = AntColonyOptimization(graph, cost, source, destination, num_iterations=num_iterations, alpha=value)
        elif parameter_name == 'beta':
            aco = AntColonyOptimization(graph, cost, source, destination, num_iterations=num_iterations, beta=value)
        elif parameter_name == 'pheromone_decay':
            aco = AntColonyOptimization(graph, cost, source, destination, num_iterations=num_iterations, pheromone_decay=value)
        elif parameter_name == 'num_ants':
            aco = AntColonyOptimization(graph, cost, source, destination, num_iterations=num_iterations, num_ants=value)
        
        path_lengths = aco.run()
        results[value] = path_lengths
    return results

def plot_results(results, parameter_name, window_size=100):
    plt.figure(figsize=(12, 6))
    for value, path_lengths in results.items():
        if len(path_lengths) < window_size:
            continue
        rolling_avg = rolling_average(path_lengths, window_size=window_size)
        x_axis = np.arange(len(rolling_avg)) + window_size
        plt.plot(x_axis, rolling_avg, label=f'{parameter_name}={value}')
    plt.xlabel('Number of Paths Found')
    plt.ylabel('Rolling Average of Path Lengths')
    plt.title(f'Influence of {parameter_name} on Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

graph = {
    0: [1, 9],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
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

cost = {
    (0, 1): 1,
    (0, 9): 1,
    (1, 0): 1,
    (1, 2): 1,
    (2, 1): 1,
    (2, 3): 1,
    (3, 2): 1,
    (3, 4): 1,
    (4, 3): 1,
    (4, 5): 1,
    (5, 4): 1,
    (5, 6): 1,
    (6, 5): 1,
    (6, 7): 1,
    (7, 6): 1,
    (7, 8): 1,
    (8, 7): 1,
    (8, 13): 1,
    (9, 0): 1,
    (9, 10): 1,
    (9, 14): 1,
    (10, 9): 1,
    (10, 11): 1,
    (10, 12): 1,
    (10, 17): 1,
    (11, 10): 1,
    (11, 12): 1,
    (12, 10): 1,
    (12, 11): 1,
    (12, 13): 1,
    (12, 18): 1,
    (13, 8): 1,
    (13, 12): 1,
    (13, 16): 1,
    (14, 9): 1,
    (14, 15): 1,
    (14, 16): 1,
    (14, 17): 1,
    (15, 14): 1,
    (15, 16): 1,
    (16, 13): 1,
    (16, 14): 1,
    (16, 15): 1,
    (16, 18): 1,
    (17, 10): 1,
    (17, 14): 1,
    (17, 18): 1,
    (18, 12): 1,
    (18, 16): 1,
    (18, 17): 1,
}

num_iterations = 1000

alpha_values = [1, 2]
beta_values = [1, 2, 3]
pheromone_decay_values = [0, 0.01, 0.1]
num_ants_values = [32, 64, 128, 256]

# alpha_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='alpha', parameter_values=alpha_values)
#beta_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='beta', parameter_values=beta_values)
pheromone_decay_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='pheromone_decay', parameter_values=pheromone_decay_values)
#num_ants_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='num_ants', parameter_values=num_ants_values)

#plot_results(alpha_results, 'alpha')
#plot_results(beta_results, 'beta')
plot_results(pheromone_decay_results, 'pheromone_decay')
#plot_results(num_ants_results, 'num_ants')