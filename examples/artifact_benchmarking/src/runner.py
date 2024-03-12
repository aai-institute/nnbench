import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from training.training import tokenize_and_align_labels
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import nnbench
from nnbench import context, reporter, types


class ConllppLoader(types.ArtifactLoader):
    def __init__(self, path: str, split: str) -> None:
        self.path = path
        self.split = split

    def load(self) -> os.PathLike[str]:
        dataset = load_dataset(self.path)
        path = dataset.cache_files[self.split][0]["filename"]
        return path


class TokenClassificationModel(types.Artifact):
    def deserialize(self) -> None:
        path = Path(self.path).resolve()
        model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(
            path, use_safetensors=True
        )
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(path)
        self._value = (model, tokenizer)


class ConllValidationData(types.Artifact):
    def deserialize(self) -> None:
        dataset = Dataset.from_file(self.path)
        label_names = dataset.features["ner_tags"].feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        self._value = dataset, id2label


def main(model_paths: list[str]) -> None:
    console_reporter = reporter.BenchmarkReporter()
    runner = nnbench.BenchmarkRunner()

    models = [
        TokenClassificationModel(loader=types.LocalArtifactLoader(path)) for path in model_paths
    ]

    conllpp = ConllValidationData(ConllppLoader("conllpp", split="validation"))
    conllpp.deserialize()

    for mod in models:
        model, tokenizer = mod.value
        dataset, id2label = conllpp.value
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_and_align_labels(tokenizer, examples),
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataloader = DataLoader(
            tokenized_dataset,
            shuffle=False,
            collate_fn=DataCollatorForTokenClassification(tokenizer, padding=True),
            batch_size=8,
        )
        result = runner.run(
            "benchmark.py",
            params={
                "model": model,
                "test_dataloader": dataloader,
                "index_label_mapping": id2label,
            },
            context=context.Context.make(
                {
                    "model": model.name_or_path,
                    "dataset": str(dataset),
                }
            ),
            tags=("metric", "model-meta"),
        )
        console_reporter.display(result, exclude=("parameters",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run named entity recognition model benchmark.")
    parser.add_argument("model_paths", type=str, nargs="+", help="Filepaths to pretrained models.")
    args = parser.parse_args()
    main(args.model_paths)
