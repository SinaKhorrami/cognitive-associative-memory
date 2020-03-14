class EpisodicMemoryLink(object):
    """ Author: Sina """

    def __init__(self, node_1, node_2):
        super().__init__()
        self.age = 0
        self.nodes = [node_1, node_2]


class SemanticMemoryLink(object):
    """ Author: Sina """

    def __init__(self, key_class, response_class):
        super().__init__()
        self.strength = 1
        self.key = key_class
        self.response = response_class
