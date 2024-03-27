import os
import tempfile
import time
from functools import cache, lru_cache, partial
from typing import Sequence

import torch
from datasets import Dataset, load_dataset
from torch.nn import Module
from torch.utils.data import DataLoader
from training import tokenize_and_align_labels
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import nnbench
from nnbench.types import Memo, cached_memo


class TokenClassificationModelMemo(Memo[Module]):
    def __init__(self, path: str):
        self.path = path

    @cached_memo
    def __call__(self) -> Module:
        model: Module = AutoModelForTokenClassification.from_pretrained(
            self.path, use_safetensors=True
        )
        return model

    def __str__(self):
        return self.path


class TokenizerMemo(Memo[PreTrainedTokenizerBase]):
    def __init__(self, path: str):
        self.path = path

    @cached_memo
    def __call__(self) -> PreTrainedTokenizerBase:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.path)
        return tokenizer

    def __str__(self):
        return self.path


class ConllValidationMemo(Memo[Dataset]):
    def __init__(self, path: str, split: str):
        self.path = path
        self.split = split

    @cached_memo
    def __call__(self) -> Dataset:
        dataset = load_dataset(self.path)
        path = dataset.cache_files[self.split][0]["filename"]
        dataset = Dataset.from_file(path)
        return dataset

    def __str__(self):
        return self.path + "/" + self.split


class IndexLabelMapMemo(Memo[dict[int, str]]):
    def __init__(self, path: str, split: str):
        self.path = path
        self.split = split

    @cached_memo
    def __call__(self) -> dict[int, str]:
        dataset = load_dataset(self.path)
        path = dataset.cache_files[self.split][0]["filename"]
        dataset = Dataset.from_file(path)
        label_names: Sequence[str] = dataset.features["ner_tags"].feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        return id2label

    def __str__(self):
        return self.path + "/" + self.split


@cache
def make_dataloader(tokenizer, dataset):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(tokenizer, examples),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return DataLoader(
        tokenized_dataset,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(tokenizer, padding=True),
        batch_size=8,
    )


@lru_cache
def run_eval_loop(model, dataloader, padding_id=-100):
    true_positives = torch.zeros(model.config.num_labels)
    false_positives = torch.zeros(model.config.num_labels)
    true_negatives = torch.zeros(model.config.num_labels)
    false_negatives = torch.zeros(model.config.num_labels)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            valid_indices = labels.view(-1) != padding_id
            predictions = predictions.view(-1)[valid_indices]
            labels = labels.view(-1)[valid_indices]
            for idx in range(model.config.num_labels):
                tp = ((predictions == idx) & (labels == idx)).sum()
                fp = ((predictions == idx) & (labels != idx)).sum()
                fn = ((predictions != idx) & (labels == idx)).sum()
                tn = ((predictions != idx) & (labels != idx)).sum()

                true_positives[idx] += tp
                false_positives[idx] += fp
                false_negatives[idx] += fn
                true_negatives[idx] += tn

    return true_positives, false_positives, true_negatives, false_negatives


parametrize_label = partial(
    nnbench.product,
    model=[TokenClassificationModelMemo("dslim/distilbert-NER")],
    tokenizer=[TokenizerMemo("dslim/distilbert-NER")],
    valdata=[ConllValidationMemo(path="conllpp", split="validation")],
    index_label_mapping=[IndexLabelMapMemo(path="conllpp", split="validation")],
    class_label=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
    tags=("metric", "per-class"),
)()


@nnbench.benchmark(tags=("metric", "aggregate"))
def precision(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)
    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    return torch.mean(precision).item()


@parametrize_label
def precision_per_class(
    class_label: str,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    precision_values = tp / (tp + fp + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return precision_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def recall(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    recall = tp / (tp + fn + 1e-6)
    return torch.mean(recall).item()


@parametrize_label
def recall_per_class(
    class_label: str,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    recall_values = tp / (tp + fn + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return recall_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def f1(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return torch.mean(f1).item()


@parametrize_label
def f1_per_class(
    class_label: str,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_values = 2 * (precision * recall) / (precision + recall + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return f1_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def accuracy(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    padding_id: int = -100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return torch.mean(accuracy).item()


@parametrize_label
def accuracy_per_class(
    class_label: str,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> dict[str, float]:
    dataloader = make_dataloader(tokenizer, valdata)

    tp, fp, tn, fn = run_eval_loop(model, dataloader, padding_id)
    accuracy_values = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return accuracy_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("config",))
def model_configuration(model: Module) -> dict:
    model.eval()
    config = model.config.to_dict()
    return config


@nnbench.benchmark(tags=("model-meta", "inference-time"))
def avg_inference_time_ns(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    valdata: Dataset,
    avg_n: int = 100,
) -> float:
    dataloader = make_dataloader(tokenizer, valdata)

    start_time = time.perf_counter()
    model.eval()
    num_datapoints = 0
    with torch.no_grad():
        for batch in dataloader:
            if num_datapoints >= avg_n:
                break
            num_datapoints += len(batch)
            _ = model(**batch)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    average_time = total_time / num_datapoints
    return average_time


@nnbench.benchmark(tags=("model-meta", "size-on-disk"))
def model_size_mb(model: Module) -> float:
    model.eval()
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        tmp.flush()
        tmp.seek(0, os.SEEK_END)
        tmp_size = tmp.tell()
    size_mb = tmp_size / (1024 * 1024)
    return size_mb
