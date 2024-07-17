import numpy as np

class AntColonyOptimization:
    """
    Class for solving the shortest path problem using the Ant Colony Optimization algorithm.
    
    Attributes:
        graph (dict): Adjacency list representation of the graph.
        source (int): The starting node for the ants.
        destination (int): The goal node for the ants.
        num_ants (int): The number of ants used in the algorithm.
        num_iterations (int): The number of iterations for the algorithm.
        pheromone_decay (float): The rate at which pheromones decay.
        alpha (float): The exponent for pheromone influence.
        pheromone (dict): Dictionary to store pheromone levels on edges.
    """
    def __init__(self, graph, source, destination, num_ants=64, num_iterations=1000, pheromone_decay=0.01, alpha=2):
        self.graph = graph # Adjacency list
        self.source = source
        self.destination = destination
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone_decay = pheromone_decay
        self.alpha = alpha
        self.pheromone = {}
        
        # Initialize pheromone levels for all edges in the graph
        for node in graph:
            for neighbor in graph[node]:
                self.pheromone[(node, neighbor)] = 1
        
    def run(self):
        """
        Executes the Ant Colony Optimization algorithm for a specified number of iterations and ants.
        
        This function repeatedly generates paths using ants and updates the pheromone levels on the paths
        taken by the ants. Pheromone levels decay after each iteration.
        """
        for _ in range(self.num_iterations):
            for _ in range(self.num_ants):
                path = self._find_path()
                if path:
                    self._deposit_pheromone(path)
            self._decay_pheromone()

    def _find_path(self):
        """
        Finds a path from the source to the destination using an ant's traversal rules.
        
        Returns:
            list: A list of nodes representing the path from source to destination, or None if no path is found.
        """
        current_node = self.source
        path = [current_node]
        visited = set()
        visited.add(current_node)
        
        while current_node != self.destination:
            next_node = self._choose_next_node(current_node, visited)
            if next_node is None:
                return None
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path

    def _choose_next_node(self, current_node, visited):
        """
        Chooses the next node for an ant to move to based on pheromone levels and visitation status.
        
        Args:
            current_node (int): The current node where the ant is located.
            visited (set): A set of visited nodes.
        
        Returns:
            int: The next node to move to, or None if no valid move is possible.
        """
        neighbors = self.graph[current_node]
        pheromones = np.array([self.pheromone[(current_node, neighbor)] for neighbor in neighbors])
        pheromones = pheromones ** self.alpha
        total_pheromones = np.sum(pheromones)
        if total_pheromones == 0:
            return None
        probabilities = pheromones / total_pheromones
        next_node = np.random.choice(neighbors, p=probabilities)
        return next_node if next_node not in visited else None

    def _decay_pheromone(self):
        """
        Decays the pheromone levels on all edges by the specified decay rate.
        """
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.pheromone_decay)

    def _deposit_pheromone(self, path):
        """
        Deposits pheromones on the edges of the given path.
        
        Args:
            path (list): The path taken by an ant, represented as a list of nodes.
        """
        path_length = len(path) - 1
        pheromone_deposit = 1 / path_length
        for i in range(path_length):
            edge = (path[i], path[i + 1])
            self.pheromone[edge] += pheromone_deposit