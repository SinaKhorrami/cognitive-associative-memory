

class EpisodicMemoryNode(object):
    """ Author: Sina """

    def __init__(self, weight, class_name, th=0, m=1):
        super().__init__()
        self.weight = weight
        self.class_name = class_name
        self.similarity_threshold = th
        self.represented_patterns_number = m


class SemanticMemoryNode(object):
    """ Author: Sina """

    def __init__(self):
        super().__init__()
