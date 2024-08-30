from sklearn.model_selection import train_test_split
from pytorch_tabular.utils import load_covertype_dataset


def covtype_data_loader():
    data, _, _, _ = load_covertype_dataset()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, valid = train_test_split(train, test_size=0.2, random_state=42)
    return train, valid, test
