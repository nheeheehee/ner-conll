import argparse

import datasets
import engine
import torch
from dataset import NERDataset
from model import NERModel
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import save_model_checkpoint, seed_everything

import config


def argument_parser(parser):
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=config.TRAIN_BATCH_SIZE
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=config.LR)
    parser.add_argument("-ep", "--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("-l", "--load", type=bool, default=False)

    args = parser.parse_args()

    return args


def download_dataset():
    data = datasets.load_dataset("conll2003")

    train_sentences = data["train"]["tokens"]
    val_sentences = data["validation"]["tokens"]
    test_sentences = data["test"]["tokens"]

    train_pos = data["train"]["pos_tags"]
    val_pos = data["validation"]["pos_tags"]
    test_pos = data["test"]["pos_tags"]

    train_tag = data["train"]["ner_tags"]
    val_tag = data["validation"]["ner_tags"]
    test_tag = data["test"]["ner_tags"]

    num_pos = len(data["train"].features["pos_tags"].feature.names)
    num_tag = len(data["train"].features["ner_tags"].feature.names)

    return (
        train_sentences,
        val_sentences,
        test_sentences,
        train_pos,
        val_pos,
        test_pos,
        train_tag,
        val_tag,
        test_tag,
        num_pos,
        num_tag,
    )


if __name__ == "__main__":
    (
        train_sentences,
        val_sentences,
        test_sentences,
        train_pos,
        val_pos,
        test_pos,
        train_tag,
        val_tag,
        test_tag,
        num_pos,
        num_tag,
    ) = download_dataset()

    train_dataset = NERDataset(texts=train_sentences, pos=train_pos, tags=train_tag)
    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, drop_last=True
    )

    val_dataset = NERDataset(texts=val_sentences, pos=val_pos, tags=val_tag)
    val_loader = DataLoader(
        val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=4, drop_last=True
    )

    parser = argparse.ArgumentParser()
    parser = argument_parser(parser)

    device = config.DEVICE

    model = NERModel(num_pos, num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    seed_everything()

    best_loss = 0.0
    loss_hist = []

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_loader, model, optimizer, scheduler)
        test_loss = engine.eval_fn(val_loader, model, optimizer, scheduler)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        loss_hist.append(test_loss)
        early_stop_count = 0
        if test_loss < best_loss:
            save_model_checkpoint(model, config.MODEL_PATH)
            best_loss = test_loss
        else:
            early_stop_count += 1
            if early_stop_count == 10:
                break
