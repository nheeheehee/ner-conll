import torch
from tqdm import tqdm

import config


def train_fn(data_loader, model, optimizer, scheduler, clip=1, device=config.DEVICE):

    model.train()
    total_loss = 0.0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
            optimizer.zero_grad()

            _, _, loss = model(**data)
            loss.backward()

            # clip gradient to prevent exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate_fn(data_loader, model, device=config.DEVICE):

    model.eval()
    total_loss = 0.0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to_device(device)

            output = model(**data)
            loss = output[0]

            total_loss += loss.item()

    return total_loss / len(data_loader)
