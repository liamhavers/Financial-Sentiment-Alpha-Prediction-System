"""
Shared fine-tuning utilities for sentiment_finbert_finetune.py and
sentiment_stocktwits_roberta_finetune.py — both fine-tune a pretrained
transformer classifier on our own labeled StockTwits data (bullish/bearish,
from messages.sentiment_label) instead of running it zero-shot as
sentiment_finbert.py / sentiment_stocktwits_roberta.py do.

CPU-only (see CLAUDE.md: no GPU available/needed at this dataset size), so
this is a plain PyTorch training loop rather than the HuggingFace Trainer/
accelerate stack, which would be overkill for a few hundred examples.
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class LabeledTextDataset(Dataset):
    """Tokenizes lazily per example; DataCollatorWithPadding pads each batch
    to its own longest member rather than a fixed MAX_LENGTH up front."""

    def __init__(self, bodies: list[str], label_ids: list[int], tokenizer, max_length: int = 128):
        self.bodies = bodies
        self.label_ids = label_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.bodies)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(self.bodies[idx], truncation=True, max_length=self.max_length)
        encoding["labels"] = self.label_ids[idx]
        return encoding


def class_weights(label_ids: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights — same 'balanced' idea as sklearn's
    class_weight='balanced' (used in sentiment_tfidf.py) — since the labeled
    set is ~4:1 bullish:bearish."""
    counts = [0] * num_classes
    for label_id in label_ids:
        counts[label_id] += 1
    n = len(label_ids)
    weights = [n / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def fine_tune(model, tokenizer, bodies: list[str], label_ids: list[int], num_classes: int,
              epochs: int = 3, batch_size: int = 16, lr: float = 2e-5) -> None:
    """Fine-tunes `model` in place on (bodies, label_ids)."""
    device = torch.device("cpu")
    model.to(device)
    model.train()

    dataset = LabeledTextDataset(bodies, label_ids, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(label_ids, num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        log.info("  epoch %d/%d: avg loss=%.4f", epoch + 1, epochs, total_loss / len(dataset))

    model.eval()


def evaluate_binary(preds: list[str], truth: list[str], model_name: str) -> float:
    """Held-out evaluation, logged the same way as sentiment_tfidf.py's
    evaluate(): classification_report + confusion matrix + accuracy."""
    labels_sorted = sorted(set(truth))
    log.info("=== %s held-out evaluation (n=%d) ===", model_name, len(truth))
    log.info("\n%s", classification_report(truth, preds, zero_division=0))
    log.info("Confusion matrix (rows=true, cols=pred, order=%s):\n%s",
              labels_sorted, confusion_matrix(truth, preds, labels=labels_sorted))
    accuracy = sum(1 for p, t in zip(preds, truth) if p == t) / len(truth)
    log.info("%s accuracy=%.3f", model_name, accuracy)
    return accuracy
