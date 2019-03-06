from src.layer import EpisodicMemoryLayer


class CognitiveAM(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
        self.episodic_memory_layer = EpisodicMemoryLayer()

    def store(self, key_vectors, response_vectors, key_classes, response_classes):
        pass

    def store_instance(self, key_vector, response_vector, key_class, response_class):
        pass

    def recall(self, key_vector):
        pass

