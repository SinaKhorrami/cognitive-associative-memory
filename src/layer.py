import numpy as np

from src.node import EpisodicMemoryNode


class EpisodicMemoryLayer(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
        self.node_set = list()
        self.subnetwork_set = dict()
        self.connection_set = list()

    def store(self, input_pattern, input_class):
        if input_class not in self.subnetwork_set:
            self._add_new_subnetwork(input_pattern=input_pattern, input_class=input_class)
            return

        winner, runner_up = self._find_winner_and_runner_up(
            input_pattern=input_pattern,
            subnetwork=self.subnetwork_set.get(input_class)
        )
        winner.represented_patterns_number += 1

        distance_to_winner = self._find_distance(first_pattern=input_pattern, second_pattern=winner.weight)
        if distance_to_winner > winner.similarity_threshold:
            node = EpisodicMemoryNode(weight=input_pattern, class_name=input_class, th=distance_to_winner, m=1)
            self.node_set.append(node)
            self.subnetwork_set[input_class].append(node)
            winner.similarity_threshold = distance_to_winner
        else:
            winner.weight = winner.weight + (input_pattern - winner.weight) / winner.represented_patterns_number
            runner_up.weight = runner_up.weight + (input_pattern - runner_up.weight) / (winner.represented_patterns_number * 100)
            winner.similarity_threshold = (winner.similarity_threshold + distance_to_winner) / 2

        # TODO: Add Connection to connection set and prune old edges

    @staticmethod
    def _find_distance(first_pattern, second_pattern):
        return np.linalg.norm(first_pattern-second_pattern)

    def _find_winner_and_runner_up(self, input_pattern, subnetwork):
        # TODO: Check Algorithm
        winner = subnetwork[0]
        min_distance = self._find_distance(input_pattern, winner.weight)

        for node in subnetwork:
            node_distance = self._find_distance(input_pattern, node.weight)
            if node_distance < min_distance:
                min_distance = node_distance
                winner = node

        runner_up = subnetwork[0]
        second_min_distance = self._find_distance(input_pattern, runner_up)

        for node in subnetwork:
            node_distance = self._find_distance(input_pattern, node.weight)
            if node_distance < second_min_distance and node != winner:
                second_min_distance = node_distance
                runner_up = node

        return winner, runner_up

    def _add_new_subnetwork(self, input_pattern, input_class):
        node = EpisodicMemoryNode(weight=input_pattern, class_name=input_class, th=0, m=1)
        self.node_set.append(node)
        self.subnetwork_set.update({
            input_class: [node]
        })


class SemanticMemoryLayer(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
