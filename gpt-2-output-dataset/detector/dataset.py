import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer



def load_texts(data_file):
    """
    Extract 'text' and 'label' from JSONL files.
    """
    texts, labels = [], []
    with open(data_file, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                texts.append(entry['text'])
                labels.append(int(entry['label']))  # Assuming label field exists
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")
            except KeyError as e:
                print(f"Key error ({e}) in line: {line}")
    return texts, labels


class Corpus:
    """
    Modified to get labels and text.
    """
    def __init__(self, data_dir='/Users/dustinhayes/Desktop/DEEP LEARNING FINAL PROJECT/gpt-2-output-dataset/detector/Data'):
        self.train_texts, self.train_labels = load_texts(f'{data_dir}/train.jsonl')
        self.test_texts, self.test_labels = load_texts(f'{data_dir}/test.jsonl')
        self.valid_texts, self.valid_labels = load_texts(f'{data_dir}/validation.jsonl')


class EncodedDataset(Dataset):
    """
    A PyTorch Dataset class for encoding text data suitable for training transformer-based models.

    This class handles the preprocessing of text data by converting raw text into a format that
    is compatible with transformer architectures, specifically RoBERTa. It tokenizes the text,
    adds necessary special tokens, and ensures that all sequences are of a uniform length via
    padding or truncation. Additionally, it generates attention masks to enable the model to
    distinguish between actual data and padding.

    The dataset pairs each text input with its corresponding label, facilitating the use of
    this dataset in supervised learning tasks, particularly classification.

    Parameters:
    texts (List[str]): A list of text strings to be tokenized and encoded.
    labels (List[int]): A list of integer labels corresponding to the texts.
    tokenizer (PreTrainedTokenizer): An instance of a tokenizer compatible with the
        transformer model being used.
    max_sequence_length (int, optional): The maximum length of the tokenized sequences.
        If not specified, defaults to the tokenizer's maximum allowed length.

    The `__getitem__` method of the dataset retrieves the tokenized representation of a text and its label, returning them as PyTorch tensors.
    This makes the class compatible with PyTorch's DataLoader for efficient batch processing during model training.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_sequence_length: int = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length or tokenizer.model_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Encode the text using the tokenizer
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        # Create tensors
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label
