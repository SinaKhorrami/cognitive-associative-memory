from src.layer import EpisodicMemoryLayer, SemanticMemoryLayer


class CognitiveAM(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
        self.episodic_memory_layer = EpisodicMemoryLayer()
        self.semantic_memory_layer = SemanticMemoryLayer()

    def store(self, key_vectors, response_vectors, key_classes, response_classes):
        pass

    def store_instance(self, key_vector, key_class, response_vector=None, response_class=None):
        key_frequent_winner = self.episodic_memory_layer.store(input_pattern=key_vector, input_class=key_class)
        self.semantic_memory_layer.update_node(input_pattern=key_vector, input_class=key_class, frequent_winner=key_frequent_winner)

        if response_vector is not None and response_class is not None:
            response_frequent_winner = self.episodic_memory_layer.store(input_pattern=response_vector, input_class=response_class)
            self.semantic_memory_layer.update_node(input_pattern=response_vector, input_class=response_class, frequent_winner=response_frequent_winner)

            self.semantic_memory_layer.update_arrow_edge(key_class=key_class, response_class=response_class)

    def recall(self, key_vector):
        winner, failed = self.episodic_memory_layer.recall(input_pattern=key_vector)

        if failed:
            pass

        completed_pattern = winner.weight
        key_class = winner.class_name

        associated_classes = self.semantic_memory_layer.recall_associated_classes(key_class_name=key_class)
        associated_patterns = self.semantic_memory_layer.recall(key_class_name=key_class, associated_classes=associated_classes)

        associated_patterns = [
            {
                'pattern': node.weight,
                'class': node.class_name
            }
            for node in associated_patterns
        ]

        return {
            'completed_pattern': completed_pattern,
            'class': key_class,
            'associated_patterns': associated_patterns
        }
