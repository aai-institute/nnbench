from collections.abc import Iterable

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

import nnbench

dataset = load_dataset("conllpp")
path = dataset.cache_files["validation"][0]["filename"]


def main() -> None:
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/distilbert-NER", use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
    valdata = Dataset.from_file(path)
    label_names: Iterable[str] = valdata.features["ner_tags"].feature.names
    index_label_mapping = {i: label for i, label in enumerate(label_names)}

    params = {
        "model": model,
        "tokenizer": tokenizer,
        "valdata": valdata,
        "index_label_mapping": index_label_mapping,
    }

    benchmarks = nnbench.collect("benchmark.py", tags=("per-class",))
    reporter = nnbench.ConsoleReporter()
    result = nnbench.run(benchmarks, params=params)
    reporter.display(result)


if __name__ == "__main__":
    main()
