from typing import Any

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    get_scheduler,
)

OUTPUT_DIR = "artifacts"


def align_labels_with_tokens(labels: list[int], word_ids: list[int]) -> list[int]:
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(
    tokenizer: PreTrainedTokenizer, examples: dict[str, Any]
) -> BatchEncoding:
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels.append(align_labels_with_tokens(label, word_ids))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_model(tokenizer_name: str, model_name: str, output_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9).to(device)

    dataset = load_dataset("conllpp", split="train")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align_labels(tokenizer, examples),
        batched=True,
        remove_columns=dataset.column_names,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    train_dataloader = DataLoader(
        tokenized_datasets,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model_save_path = f"{output_dir}/{model_name}"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    tokenizers_and_models = [
        ("distilbert-base-uncased", "distilbert-base-uncased"),
        ("bert-base-uncased", "bert-base-uncased"),
    ]
    for t, m in tokenizers_and_models:
        train_model(t, m, OUTPUT_DIR)
