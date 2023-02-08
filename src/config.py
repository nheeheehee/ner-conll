from pathlib import Path

import torch
import transformers

MAIN_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = MAIN_PATH / "artifact" / "model_checkpoint"
DATA_PATH = MAIN_PATH / "data"
TRAINING_FILE = DATA_PATH / "train.txt"

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
BASE_MODEL = "bert-base-uncased"
