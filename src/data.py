import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


def download_data():
    file_path = "pg145.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    sampler=None,
):

    text_data = download_data()

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_data = text_data[:split_idx]
    eval_data = text_data[split_idx:]

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    train_dataset = GPTDatasetV1(train_data, tokenizer, max_length, stride)
    eval_dataset = GPTDatasetV1(eval_data, tokenizer, max_length, stride)

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=None if sampler is None else sampler(train_dataset),
        shuffle=shuffle and sampler is None,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,  # ! optimized step 4: pin memory to GPU
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=None if sampler is None else sampler(eval_dataset),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,  # ! optimized step 4: pin memory to GPU
    )

    return train_dataloader, eval_dataloader
