import numpy as np

from src.node import EpisodicMemoryNode, SemanticMemoryNode
from src.link import EpisodicMemoryLink, SemanticMemoryLink


class EpisodicMemoryLayer(object):
    """ Author: Sina """

    def __init__(self, max_age=10):
        super().__init__()
        self.node_set = list()
        self.subnetwork_set = dict()
        self.connection_set = list()
        self.max_age = max_age

    def store(self, input_pattern, input_class):
        if input_class not in self.subnetwork_set:
            self._add_new_subnetwork(input_pattern=input_pattern, input_class=input_class)
            return self._get_subnetwork_frequent_winner(input_class=input_class)

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
            winner.similarity_threshold = (winner.similarity_threshold + distance_to_winner) / 2
            if runner_up is not None:
                runner_up.weight = runner_up.weight + (input_pattern - runner_up.weight) / (winner.represented_patterns_number * 100)

        if runner_up is None:
            return self._get_subnetwork_frequent_winner(input_class=input_class)

        connection = None
        for c in self.connection_set:
            if winner in c.nodes and runner_up in c.nodes:
                connection = c
                break
        if connection is None:
            connection = EpisodicMemoryLink(node_1=winner, node_2=runner_up)

        for c in self.connection_set:
            if winner in c.nodes and runner_up not in c.nodes:
                c.age += 1

        self.connection_set.append(connection)

        return self._get_subnetwork_frequent_winner(input_class=input_class)

    def remove_old_edges(self):
        self.connection_set = [c for c in self.connection_set if c.age <= self.max_age]

    def remove_isolated_nodes(self):
        pass

    @staticmethod
    def _find_distance(first_pattern, second_pattern):
        return np.linalg.norm(first_pattern-second_pattern)

    def _find_winner_and_runner_up(self, input_pattern, subnetwork):
        if len(subnetwork) == 1:
            return subnetwork[0], None

        winner = subnetwork[0]
        min_distance = self._find_distance(input_pattern, winner.weight)

        for node in subnetwork:
            node_distance = self._find_distance(input_pattern, node.weight)
            if node_distance < min_distance:
                min_distance = node_distance
                winner = node

        runner_up = subnetwork[0] if subnetwork[0] != winner else subnetwork[1]
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

    def _get_subnetwork_frequent_winner(self, input_class):
        subnetwork = self.subnetwork_set.get(input_class)
        frequent_winner = subnetwork[0]
        for node in subnetwork[1:]:
            if node.represented_patterns_number > frequent_winner.represented_patterns_number:
                frequent_winner = node

        return frequent_winner

    def recall(self, input_pattern):
        min_distance_node = self.node_set[0]
        min_distance = self._find_distance(first_pattern=input_pattern, second_pattern=min_distance_node.weight)
        min_distance = min_distance*min_distance
        for node in self.node_set[1:]:
            distance = self._find_distance(first_pattern=input_pattern, second_pattern=node.weight)
            distance = distance*distance
            if distance < min_distance:
                min_distance = distance
                min_distance_node = node
        if min_distance > min_distance_node.similarity_threshold:
            return min_distance_node, True

        return min_distance_node, False


class SemanticMemoryLayer(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
        self.node_set = list()
        self.arrow_edge_set = list()

    def update_node(self, input_pattern, input_class, frequent_winner):
        insert = True
        for node in self.node_set:
            if input_class == node.class_name:
                insert = False
                node.weight = frequent_winner
                break
        if insert:
            node = SemanticMemoryNode(weight=input_pattern, class_name=input_class)
            self.node_set.append(node)

    def update_arrow_edge(self, key_class, response_class):
        insert = True
        for arrow_edge in self.arrow_edge_set:
            if arrow_edge.key == key_class and arrow_edge.response == response_class:
                insert = False
                arrow_edge.strength += 1
                break
        if insert:
            link = SemanticMemoryLink(key_class=key_class, response_class=response_class)
            self.arrow_edge_set.append(link)

    def recall_associated_classes(self, key_class_name):
        associated_classes = list()

        for link in self.arrow_edge_set:
            if link.key == key_class_name:
                associated_classes.append(link)

        associated_classes.sort(key=lambda n: n.strength, reverse=True)
        associated_classes = [item.response for item in associated_classes]
        return associated_classes

    def recall(self, key_class_name, associated_classes=None):
        if associated_classes is None:
            associated_classes = self.recall_associated_classes(key_class_name=key_class_name)

        associated_patterns = [None for _ in associated_classes]

        for node in self.node_set:
            if node.class_name in associated_classes:
                index = associated_classes.index(node.class_name)
                associated_patterns[index] = node

        return associated_patterns
