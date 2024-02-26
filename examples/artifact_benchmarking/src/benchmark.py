import os
import tempfile
import time
from functools import lru_cache

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import nnbench


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


@nnbench.benchmark(tags=("metric", "aggregate"))
def precision(model: Module, test_dataloader: DataLoader, padding_id: int = -100) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    return torch.mean(precision).item()


@nnbench.parametrize(
    (
        {"class_label": "O"},
        {"class_label": "B-PER"},
        {"class_label": "I-PER"},
        {"class_label": "B-ORG"},
        {"class_label": "I-ORG"},
        {"class_label": "B-LOC"},
        {"class_label": "I-LOC"},
        {"class_label": "B-MISC"},
        {"class_label": "I-MISC"},
    ),
    tags=("metric", "per-class"),
)
def precision_per_class(
    class_label: str,
    model: Module,
    test_dataloader: DataLoader,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    precision_values = tp / (tp + fp + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return precision_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def recall(model: Module, test_dataloader: DataLoader, padding_id: int = -100) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    recall = tp / (tp + fn + 1e-6)
    return torch.mean(recall).item()


@nnbench.parametrize(
    (
        {"class_label": "O"},
        {"class_label": "B-PER"},
        {"class_label": "I-PER"},
        {"class_label": "B-ORG"},
        {"class_label": "I-ORG"},
        {"class_label": "B-LOC"},
        {"class_label": "I-LOC"},
        {"class_label": "B-MISC"},
        {"class_label": "I-MISC"},
    ),
    tags=("metric", "per-class"),
)
def recall_per_class(
    class_label: str,
    model: Module,
    test_dataloader: DataLoader,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    recall_values = tp / (tp + fn + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return recall_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def f1(model: Module, test_dataloader: DataLoader, padding_id: int = -100) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return torch.mean(f1).item()


@nnbench.parametrize(
    (
        {"class_label": "O"},
        {"class_label": "B-PER"},
        {"class_label": "I-PER"},
        {"class_label": "B-ORG"},
        {"class_label": "I-ORG"},
        {"class_label": "B-LOC"},
        {"class_label": "I-LOC"},
        {"class_label": "B-MISC"},
        {"class_label": "I-MISC"},
    ),
    tags=("metric", "per-class"),
)
def f1_per_class(
    class_label: str,
    model: Module,
    test_dataloader: DataLoader,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_values = 2 * (precision * recall) / (precision + recall + 1e-6)
    for idx, lbl in index_label_mapping.items():
        if lbl == class_label:
            return f1_values[idx]
    raise ValueError(f" Key {class_label} not in test labels")


@nnbench.benchmark(tags=("metric", "aggregate"))
def accuracy(model: Module, test_dataloader: DataLoader, padding_id: int = -100) -> float:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return torch.mean(accuracy).item()


@nnbench.parametrize(
    (
        {"class_label": "O"},
        {"class_label": "B-PER"},
        {"class_label": "I-PER"},
        {"class_label": "B-ORG"},
        {"class_label": "I-ORG"},
        {"class_label": "B-LOC"},
        {"class_label": "I-LOC"},
        {"class_label": "B-MISC"},
        {"class_label": "I-MISC"},
    ),
    tags=("metric", "per-class"),
)
def accuracy_per_class(
    class_label: str,
    model: Module,
    test_dataloader: DataLoader,
    index_label_mapping: dict[int, str],
    padding_id: int = -100,
) -> dict[str, float]:
    tp, fp, tn, fn = run_eval_loop(model, test_dataloader, padding_id)
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
def avg_inference_time_ns(model: Module, test_dataloader: DataLoader, avg_n: int = 100) -> float:
    start_time = time.perf_counter()
    model.eval()
    num_datapoints = 0
    with torch.no_grad():
        for batch in test_dataloader:
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
