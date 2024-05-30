import os
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader


def create_segment_embedding(char_list, max_len) -> torch.Tensor:
    """
    Create segment embeddings for a list of characters with a given maximum length.

    Args:
        char_list (List[str]): List of characters.
        max_len (int): Maximum length of the segment embedding.

    Returns:
        torch.Tensor: Tensor containing segment embeddings.
    """
    segment_ids = []

    comma_indices: list[int] = [i for i, char in enumerate(char_list) if char == ',']

    current_segment = 0
    for i, char in enumerate(char_list):
        segment_ids.append(current_segment)
        if i in comma_indices:
            current_segment = 1

    # Ensure the length of segment_ids matches max_len
    if len(segment_ids) < max_len:
        segment_ids += [0] * (max_len - len(segment_ids))
    else:
        segment_ids = segment_ids[:max_len]

    segment_ids = torch.tensor(segment_ids)
    return segment_ids


class BertNERDataset(Dataset):
    """
    Dataset class for BERT-based Named Entity Recognition (NER).

    Args:
        texts (List[List[str]]): List of tokenized texts.
        labels (List[List[str]]): List of corresponding labels for the texts.
        vocab (Dict[str, int]): Vocabulary mapping tokens to indices.
        max_len (int): Maximum length of input sequences.
        label_map (Dict[str, int]): Mapping from label names to indices.
    """
    def __init__(self, texts, labels, vocab, max_len, label_map):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = [self.label_map[label] for label in self.labels[idx]]
        segment_embedding = create_segment_embedding(text, self.max_len)

        input_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in text]
        input_ids = (
            input_ids + [self.vocab["<PAD>"]] * (self.max_len - len(input_ids))
            if len(input_ids) < self.max_len
            else input_ids[: self.max_len]
        )
        label = (
            label + [0] * (self.max_len - len(label))
            if len(label) < self.max_len
            else label[: self.max_len]
        )

        input_ids = torch.tensor(input_ids)
        label = torch.tensor(label)
        attention_mask = (input_ids != self.vocab["<PAD>"]).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "segment_embedding": segment_embedding
        }


def read_data(text_file, label_file) -> tuple[list[list[str]], list[list[str]]]:
    """
    Read text and label data from files.

    Args:
        text_file (str): Path to the file containing texts.
        label_file (str): Path to the file containing labels.

    Returns:
        tuple: A tuple containing two lists: texts and labels.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip().split() for line in f]
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip().split() for line in f]
    return texts, labels


def create_mask(src, pad_token):
    """
    Create an attention mask for padded sequences.

    Args:
        src (torch.Tensor): Input tensor.
        pad_token (int): Padding token index.

    Returns:
        torch.Tensor: Attention mask tensor.
    """
    return (src != pad_token).unsqueeze(1).unsqueeze(2)


def get_train_dev_dataloader(dataset_path: str, model_type: str, max_len: int = 512, batch_size: int = 2) -> tuple[DataLoader, DataLoader, dict, dict]:
    """
    Create dataloaders for training and development datasets.

    Args:
        dataset_path (str): Path to the dataset.
        model_type (str): Type of model (e.g., "encoder_only", "decoder_only").
        max_len (int, optional): Maximum length of input sequences. Defaults to 512.
        batch_size (int, optional): Batch size for dataloaders. Defaults to 2.

    Returns:
        tuple: A tuple containing train and dev dataloaders, label to ID mapping, and vocabulary.
    """
    train_texts, train_labels = read_data(os.path.join(dataset_path, "train.txt"), os.path.join(dataset_path, "train_TAG.txt"))
    dev_texts, dev_labels = read_data(os.path.join(dataset_path, "dev.txt"), os.path.join(dataset_path, "dev_TAG.txt"))

    unique_labels = {}
    for label_list in train_labels + dev_labels:
        for label in label_list:
            if label not in unique_labels:
                unique_labels[label] = True

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    # id2label = {idx: label for label, idx in label2id.items()}

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for token in text:
            if token not in vocab:
                vocab[token] = len(vocab)

    print(f"Vocab size: {len(vocab)}")

    if model_type == "encoder_only":
        train_dataset = BertNERDataset(train_texts, train_labels, vocab, max_len, label2id)
        dev_dataset = BertNERDataset(dev_texts, dev_labels, vocab, max_len, label2id)
    elif model_type == "decoder_only":
        train_dataset = BertNERDataset(train_texts, train_labels, vocab, max_len, label2id)
        dev_dataset = BertNERDataset(dev_texts, dev_labels, vocab, max_len, label2id)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, dev_data_loader, label2id, vocab


def get_test_dataloader(text_file: str, vocab: dict, label_map: dict, max_len: int = 512, batch_size: int = 2) -> DataLoader:
    """
    Create dataloader for test dataset.

    Args:
        text_file (str): Path to the file containing test texts.
        vocab (dict): Vocabulary mapping tokens to indices.
        label_map (dict): Mapping from label names to indices.
        max_len (int, optional): Maximum length of input sequences. Defaults to 512.
        batch_size (int, optional): Batch size for dataloader. Defaults to 2.

    Returns:
        DataLoader: Dataloader for the test dataset.
    """
    with open(os.path.join(text_file, "test.txt"), "r", encoding="utf-8") as f:
        texts: list[list[str]] = [line.strip().split() for line in f]
        labels: list[list[str]] = [["O"] * len(text) for text in texts]

        test_dataset = BertNERDataset(texts, labels, vocab, max_len, label_map)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_data_loader


if __name__ == "__main__":
    train_data_loader, dev_data_loader, _, _ = get_train_dev_dataloader("./ner-data", "encoder_only")
    next(iter(train_data_loader))
