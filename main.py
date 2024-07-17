import matplotlib.pyplot as plt
import numpy as np
from AntColonyOptimization import AntColonyOptimization
from collections import defaultdict

def compute_rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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

def plot_results(results, parameter_name):
    plt.figure(figsize=(12, 6))
    for value, path_lengths in results.items():
        rolling_avg = compute_rolling_average(path_lengths, window_size=50)
        plt.plot(rolling_avg, label=f'{parameter_name}={value}')
    plt.xlabel('Iterations')
    plt.ylabel('Rolling Average of Mean Path Length')
    plt.title(f'Influence of {parameter_name} on Convergence')
    plt.legend()
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
    (0, 1): 2,
    (0, 9): 4,
    (1, 0): 2,
    (1, 2): 3,
    (2, 1): 3,
    (2, 3): 1,
    (3, 2): 1,
    (3, 4): 2,
    (4, 3): 2,
    (4, 5): 2,
    (5, 4): 2,
    (5, 6): 3,
    (6, 5): 3,
    (6, 7): 2,
    (7, 6): 2,
    (7, 8): 4,
    (8, 7): 4,
    (8, 13): 6,
    (9, 0): 4,
    (9, 10): 1,
    (9, 14): 3,
    (10, 9): 1,
    (10, 11): 2,
    (10, 12): 4,
    (10, 17): 3,
    (11, 10): 2,
    (11, 12): 1,
    (12, 10): 4,
    (12, 11): 1,
    (12, 13): 3,
    (12, 18): 5,
    (13, 8): 6,
    (13, 12): 3,
    (13, 16): 4,
    (14, 9): 3,
    (14, 15): 2,
    (14, 16): 3,
    (14, 17): 1,
    (15, 14): 2,
    (15, 16): 1,
    (16, 13): 4,
    (16, 14): 3,
    (16, 15): 1,
    (16, 18): 2,
    (17, 10): 3,
    (17, 14): 1,
    (17, 18): 4,
    (18, 12): 5,
    (18, 16): 2,
    (18, 17): 4,
}

num_iterations = 500

alpha_values = [1, 2, 3]
beta_values = [1, 2, 3]
pheromone_decay_values = [0, 0.01, 0.1]
num_ants_values = [32, 64, 128, 256]

# Run simulations and plot results for alpha
alpha_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='alpha', parameter_values=alpha_values)
beta_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='beta', parameter_values=beta_values)
pheromone_decay_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='pheromone_decay', parameter_values=pheromone_decay_values)
num_ants_results = run_simulations(graph, cost, source=0, destination=8, num_iterations=num_iterations, parameter_name='num_ants', parameter_values=num_ants_values)

plot_results(alpha_results, 'alpha')
plot_results(beta_results, 'beta')
plot_results(pheromone_decay_results, 'pheromone_decay')
plot_results(num_ants_results, 'num_ants')