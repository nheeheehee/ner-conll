import os
import random

import datasets
import numpy as np
import torch

import config


def download_dataset(data_dir):
    data = datasets.load_dataset("conll2003")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_set = data["train"]
    val_set = data["validation"]
    test_set = data["test"]

    train_set = process_dataset(train_set)
    val_set = process_dataset(val_set)
    test_set = process_dataset(test_set)

    save_dataset(train_set, data_dir, "train")
    save_dataset(val_set, data_dir, "validation")
    save_dataset(test_set, data_dir, "test")

    return train_set, val_set, test_set


def process_dataset(dataset, separator="\t"):
    """Create string tabular representation of the dataset
    in the form of
    SEQ_LEN [separator] TOKENS [separator] NER_TAGS

    Args:
        dataset (_type_): _description_
        separator (str, optional): _description_. Defaults to "\t".
    """
    processed_data = []

    for sample in dataset:
        tokens = sample["tokens"]
        tokens = [token.strip() for token in tokens]

        ner_tags = sample["ner_tags"]
        ner_tags = [str(tag) for tag in ner_tags]

        sample_size = len(sample["tokens"])

        processed_sample = f"{sample_size}{separator}"
        processed_sample += separator.join(tokens + ner_tags) + "\n"
        processed_data.append(processed_sample)

    return processed_data


def save_dataset(dataset, save_dir, dataset_type):

    if dataset_type not in ["train", "validation", "test"]:
        raise ValueError("Invalid dataset. Expect: train, validation or test")

    save_path = os.path.join(save_dir, f"{dataset_type}.txt")

    with open(save_path, "w") as f:
        f.writelines(dataset)


def save_model_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print("Model successfully saved at", path)


def load_model_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))


def seed_everything(seed: int = 100):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    download_dataset(config.DATA_PATH)
