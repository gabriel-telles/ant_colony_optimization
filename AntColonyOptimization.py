import numpy as np

class AntColonyOptimization:
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
        self.ants = [{'path': [self.source], 'returning': False, 'flag': False, 'path_length': 0} for _ in range(self.num_ants)]
        
        self.pheromone = {}
        for node in graph:
            for neighbor in graph[node]:
                self.pheromone[(node, neighbor)] = 1
        
    def run(self):
        path_lengths = []
        
        for _ in range(self.num_iterations):
            for ant in self.ants:
                if not ant['returning']:
                    self._move_ant_forward(ant)
                elif not ant['flag']:
                    ant['flag'] = True
                    self._prepare_ant_for_return(ant)
                    self._move_ant_back(ant)
                else:
                    self._move_ant_back(ant)
                    
                if ant['returning'] and ant['path'][-1] == self.source:
                    path_lengths.append(ant['path_length'])
                    self._reset_ant(ant)
            
            self._decay_pheromone()
        
        return path_lengths

    def _move_ant_forward(self, ant):
        current_node = ant['path'][-1]
        next_node = self._choose_next_node(current_node, ant['path'][-2] if len(ant['path']) > 1 else None)
        if next_node:
            ant['path'].append(next_node)
            if next_node == self.destination:
                ant['returning'] = True
        else:
            ant['path'].append(ant['path'][-2])
    
    def _move_ant_back(self, ant):
        if len(ant['path']) <= 1:
            return
        current_node = ant['path'][-1]
        previous_node = ant['path'][-2]
        self._deposit_pheromone(edge=(current_node, previous_node), path_length=ant['path_length'])
        ant['path'].pop()
            
    def _prepare_ant_for_return(self, ant):
        ant['path'] = self._remove_cycles(ant['path'])
        ant['path_length'] = self._calculate_path_length(ant['path'])
    
    def _choose_next_node(self, current_node, previous_node=None):
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
        return next_node

    def _decay_pheromone(self):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.pheromone_decay)

    def _deposit_pheromone(self, edge, path_length):
        if path_length == 0:
            return
        
        pheromone_deposit = 1 / path_length
        self.pheromone[edge] += pheromone_deposit

    def _calculate_path_length(self, path):
        return sum(self.cost[(path[i], path[i + 1])] for i in range(len(path) - 1))

    def _reset_ant(self, ant):
        ant['path'] = [self.source]
        ant['returning'] = False
        ant['flag'] = False
        ant['path_length'] = 0

    def _remove_cycles(self, path):
        i = 0
        while i < len(path):
            node = path[i]
            if node in path[i + 1:]:
                j = path.index(node, i + 1)
                if j > i:
                    del path[i + 1:j + 1]
            else:
                i += 1

        return path
