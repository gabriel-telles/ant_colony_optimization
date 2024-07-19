import numpy as np

class AntColonyOptimization:
    """
    Ant Colony Optimization algorithm for finding the shortest path in a graph.

    Attributes:
        graph (dict): Adjacency list representing the graph.
        cost (dict): Dictionary representing the cost of edges.
        source (any): The starting node.
        destination (any): The destination node.
        num_ants (int): Number of ants used in the simulation.
        num_iterations (int): Number of iterations for the simulation.
        pheromone_decay (float): Rate of pheromone decay.
        alpha (float): Influence of pheromone.
        beta (float): Influence of heuristic information.
        ants (list): List of ants, each represented as a dictionary.
        pheromone (dict): Dictionary representing the pheromone levels on edges.
    """

    def __init__(self, graph, cost, source, destination, num_ants=128, num_iterations=1000, pheromone_decay=0.01, alpha=1, beta=1):
        """
        Initializes the Ant Colony Optimization algorithm.

        Args:
            graph (dict): Adjacency list representing the graph.
            cost (dict): Dictionary representing the cost of edges.
            source (any): The starting node.
            destination (any): The destination node.
            num_ants (int, optional): Number of ants used in the simulation. Default is 128.
            num_iterations (int, optional): Number of iterations for the simulation. Default is 1000.
            pheromone_decay (float, optional): Rate of pheromone decay. Default is 0.01.
            alpha (float, optional): Influence of pheromone. Default is 1.
            beta (float, optional): Influence of heuristic information. Default is 1.
        """
        self.graph = graph
        self.cost = cost
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
        """
        Runs the Ant Colony Optimization algorithm.

        Returns:
            list: List of path lengths for each ant that completes a round trip.
        """
        path_lengths = []

        for it in range(self.num_iterations):
            print(it)
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
        """
        Moves the ant forward to the next node.

        Args:
            ant (dict): The ant to move.
        """
        current_node = ant['path'][-1]
        next_node = self._choose_next_node(current_node, ant['path'][-2] if len(ant['path']) > 1 else None)
        if next_node:
            ant['path'].append(next_node)
            if next_node == self.destination:
                ant['returning'] = True
        else:
            ant['path'].append(ant['path'][-2])

    def _move_ant_back(self, ant):
        """
        Moves the ant back to the source node, depositing pheromone along the path.

        Args:
            ant (dict): The ant to move.
        """
        if len(ant['path']) > 1:
            current_node = ant['path'].pop()
            previous_node = ant['path'][-1]
            self._deposit_pheromone((previous_node, current_node), ant['path_length'])

    def _prepare_ant_for_return(self, ant):
        """
        Prepares the ant for the return trip by removing cycles from its path and calculating the path length.

        Args:
            ant (dict): The ant to prepare.
        """
        ant['path'] = self._remove_cycles(ant['path'])
        ant['path_length'] = self._calculate_path_length(ant['path'])

    def _choose_next_node(self, current_node, previous_node=None):
        """
        Chooses the next node for the ant to move to based on pheromone levels and heuristic information.

        Args:
            current_node (any): The current node.
            previous_node (any, optional): The previous node to avoid returning to. Default is None.

        Returns:
            any: The chosen next node.
        """
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
        """
        Decays the pheromone levels on all edges.
        """
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.pheromone_decay)

    def _deposit_pheromone(self, edge, path_length):
        """
        Deposits pheromone on an edge based on the path length.

        Args:
            edge (tuple): The edge to deposit pheromone on.
            path_length (float): The length of the path the ant traveled.
        """
        if path_length == 0:
            return

        pheromone_deposit = 1 / path_length
        self.pheromone[edge] += pheromone_deposit

    def _calculate_path_length(self, path):
        """
        Calculates the length of a given path.

        Args:
            path (list): The path to calculate the length of.

        Returns:
            float: The length of the path.
        """
        return sum(self.cost[(path[i], path[i + 1])] for i in range(len(path) - 1))

    def _reset_ant(self, ant):
        """
        Resets an ant to its initial state.

        Args:
            ant (dict): The ant to reset.
        """
        ant['path'] = [self.source]
        ant['returning'] = False
        ant['flag'] = False
        ant['path_length'] = 0

    def _remove_cycles(self, path):
        """
        Removes cycles from a given path.

        Args:
            path (list): The path to remove cycles from.

        Returns:
            list: The path with cycles removed.
        """
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
