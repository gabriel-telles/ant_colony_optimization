import numpy as np

class AntColonyOptimization:
    """
    Class for solving the shortest path problem using the Ant Colony Optimization algorithm.
    
    Attributes:
        graph (dict): Adjacency list representation of the graph.
        cost (dict): Dictionary to store costs associated with edges.
        source (int): The starting node for the ants.
        destination (int): The goal node for the ants.
        num_ants (int): The number of ants used in the algorithm.
        num_iterations (int): The number of iterations for the algorithm.
        pheromone_decay (float): The rate at which pheromones decay.
        alpha (float): The exponent for pheromone influence.
        beta (float): The exponent for heuristic influence.
        pheromone (dict): Dictionary to store pheromone levels on edges.
    """
    def __init__(self, graph, cost, source, destination, num_ants=128, num_iterations=1000, pheromone_decay=0.01, alpha=2, beta=1):
        self.graph = graph  # Adjacency list
        self.cost = cost  # Cost dictionary
        self.source = source
        self.destination = destination
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone_decay = pheromone_decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = {}
        
        # Initialize pheromone levels for all edges in the graph
        for node in graph:
            for neighbor in graph[node]:
                self.pheromone[(node, neighbor)] = 1
                
        # Initialize ants
        self.ants = [{'path': [self.source], 'visited': set([self.source]), 'returning': False, 'path_length': 0} for _ in range(self.num_ants)]
        
    def run(self):
        
        all_path_lengths = []
        
        for _ in range(self.num_iterations):
            path_lengths = []
            
            for ant in self.ants:
                if not ant['returning']:
                    self._move_ant_forward(ant)
                else:
                    self._move_ant_back(ant)
                    
                if ant['returning'] and ant['path'][-1] == self.source:
                    path_lengths.append(ant['path_length'])
                    self._deposit_pheromone(ant['path'])
                    self._reset_ant(ant)
                    
            if path_lengths:
                all_path_lengths.append(np.mean(path_lengths))
            self._decay_pheromone()
        
        return all_path_lengths

    def _move_ant_forward(self, ant):
        current_node = ant['path'][-1]
        if current_node == self.destination:
            ant['returning'] = True
        else:
            next_node = self._choose_next_node(current_node, ant['visited'], ant['path'][-2] if len(ant['path']) > 1 else None)
            if next_node:
                ant['path'].append(next_node)
                ant['visited'].add(next_node)
                if next_node == self.destination:
                    ant['returning'] = True
            else:
                # Reset the ant if no moves are possible and not at the destination
                self._reset_ant(ant)
    
    def _move_ant_back(self, ant):
        if len(ant['path']) > 1:
            current_node = ant['path'][-1]
            previous_node = ant['path'][-2]
            ant['path'].pop()
            ant['path_length'] += self.cost[(previous_node, current_node)]
    
    def _choose_next_node(self, current_node, visited, previous_node=None):
        neighbors = self.graph[current_node]
        
        if previous_node and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != previous_node]
        
        pheromones = np.array([self.pheromone[(current_node, neighbor)] for neighbor in neighbors])
        heuristics = np.array([1 / self.cost[(current_node, neighbor)] for neighbor in neighbors])
        
        combined_influence = (pheromones ** self.alpha) * (heuristics ** self.beta)
        
        total_influence = np.sum(combined_influence)
        if total_influence == 0:
            return previous_node if previous_node else None
        probabilities = combined_influence / total_influence
        
        next_node = np.random.choice(neighbors, p=probabilities)
        return next_node if next_node not in visited else None

    def _decay_pheromone(self):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.pheromone_decay)

    def _deposit_pheromone(self, path):
        path_length = self._calculate_path_length(path)
        if path_length == 0:
            return
        
        pheromone_deposit = 1 / path_length
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            self.pheromone[edge] += pheromone_deposit

    def _calculate_path_length(self, path):
        return sum(self.cost[(path[i], path[i + 1])] for i in range(len(path) - 1))

    def _reset_ant(self, ant):
        ant['path'] = [self.source]
        ant['visited'] = set([self.source])
        ant['returning'] = False
        ant['path_length'] = 0
