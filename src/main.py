from src.model import CognitiveAM
from src.util import get_train_charset, get_validation_charset

if __name__ == '__main__':
    model = CognitiveAM()

    train = get_train_charset()
    validation = get_validation_charset(count=10, noise=0.2)

    for train_item in train:
        model.store_instance(
            key_vector=train_item.get('key_pattern'),
            key_class=train_item.get('key_class'),
            response_vector=train_item.get('response_pattern'),
            response_class=train_item.get('response_class')
        )

    classify = list()
    # val = list()

    for validation_item in validation:
        recalled_item = model.recall(
            key_vector=validation_item.get('key_pattern')
        )
        resp = 1 if validation_item.get('key_class') == recalled_item.get('class') else 0
        classify.append(resp)

    print("CLASSIFICATION ACCURACY: {}".format(sum(classify) / len(classify)))



