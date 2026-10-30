"""Fine-tune DistilBERT with soft labels and identity weights on the Jigsaw split.

Same pipeline as train_distilbert.py, with two changes in the loss:
- soft labels: the loss compares the predicted probability with `target` (fraction of raters)
  instead of the 0/1 label;
- identity weights: comments that mention one of the 9 identities count more in the loss.
Validation metrics are still computed against the 0/1 label, so results compare with the base model.

Usage:
    python src/train_distilbert_soft.py                              # values from configs/soft.yaml
    python src/train_distilbert_soft.py --identity_weights false     # soft labels only
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from scipy.special import softmax
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/jigsaw_clean.parquet"

MAIN_IDENTITIES = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish",
    "muslim", "black", "white", "psychiatric_or_mental_illness",
]


# Configuration: soft.yaml, then command line overrides

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/soft.yaml")
    args, overrides = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Each override is "--key value"; yaml parses the value into the right type
    for key, value in zip(overrides[::2], overrides[1::2]):
        key = key.lstrip("-")
        if key not in cfg:
            raise ValueError(f"Unknown config key: {key}")
        parsed = yaml.safe_load(value)
        # yaml reads "2e-5" as a string, so numbers in that form are converted here
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        cfg[key] = parsed

    mode = ("soft" if cfg["soft_labels"] else "hard") + ("_idw" if cfg["identity_weights"] else "")
    size = f"_n{cfg['train_size']}" if cfg["train_size"] else ""
    cfg["run_name"] = f"{mode}_lr{cfg['learning_rate']:g}_ep{cfg['epochs']}{size}"
    return cfg


cfg = load_config()
run_dir = ROOT / "models" / cfg["run_name"]
metrics_dir = ROOT / "results/metrics" / cfg["run_name"]
predictions_dir = ROOT / "results/predictions" / cfg["run_name"]
for d in (metrics_dir, predictions_dir):
    d.mkdir(parents=True, exist_ok=True)

print(f"Run: {cfg['run_name']}")
print(yaml.safe_dump(cfg, sort_keys=False))


# Phase 1. Load the tokenizer and the model, and define the labels

id2label = {0: "NON-TOXIC", 1: "TOXIC"}
label2id = {"NON-TOXIC": 0, "TOXIC": 1}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(
    cfg["model_name"],
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)


# Phase 2. Build the DatasetDict, with the soft label and the loss weight of each comment

df = pd.read_parquet(DATA_PATH, columns=["id", "comment_text", "target", "label", "split"] + MAIN_IDENTITIES)

# Training target: fraction of raters (soft) or the 0/1 label (hard)
df["soft_label"] = df["target"] if cfg["soft_labels"] else df["label"].astype(float)

# Loss weight: 1 by default, higher for comments that mention an identity (value >= 0.5)
mentions_identity = (df[MAIN_IDENTITIES] >= 0.5).any(axis=1)
df["weight"] = 1.0
if cfg["identity_weights"]:
    df.loc[mentions_identity & (df["label"] == 1), "weight"] = cfg["weight_identity_toxic"]
    df.loc[mentions_identity & (df["label"] == 0), "weight"] = cfg["weight_identity_nontoxic"]

columns = ["id", "comment_text", "label", "soft_label", "weight"]


def to_dataset(name):
    return Dataset.from_pandas(df.loc[df["split"] == name, columns], preserve_index=False)


dataset = DatasetDict({
    "train": to_dataset("train"),
    "validation": to_dataset("validation"),
    "test": to_dataset("test"),
})

if cfg["train_size"]:
    dataset["train"] = dataset["train"].shuffle(seed=cfg["seed"]).select(range(cfg["train_size"]))

for name in ["train", "validation"]:
    labels = dataset[name]["label"]
    print(f"{name}: {len(labels):,} comments, {sum(labels) / len(labels) * 100:.2f}% toxic")
weights = pd.Series(dataset["train"]["weight"])
print(f"Train loss weights: {weights.value_counts().sort_index().to_dict()}")


# Phase 3. Tokenize

def preprocess_function(examples):
    return tokenizer(examples["comment_text"], truncation=True, max_length=cfg["max_length"])


# Text and id are dropped after tokenizing, so only numbers reach the model
tokenized = DatasetDict({
    name: dataset[name].map(preprocess_function, batched=True, num_proc=4, remove_columns=["id", "comment_text"])
    for name in ["train", "validation"]
})


# Phase 4. Evaluation metrics (against the 0/1 label, as in the base model)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = softmax(logits, axis=1)[:, 1]
    return {
        "auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "f1": f1_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
    }


# Phase 5. Training settings and Trainer with the weighted soft-label loss

class SoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        target = inputs.pop("soft_label")
        weight = inputs.pop("weight")
        inputs.pop("labels")
        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=1)
        # Cross-entropy against the soft target, weighted per comment
        loss = -(target * log_probs[:, 1] + (1 - target) * log_probs[:, 0])
        loss = (loss * weight).sum() / weight.sum()
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=str(run_dir),
    learning_rate=cfg["learning_rate"],
    per_device_train_batch_size=cfg["batch_size"],
    per_device_eval_batch_size=cfg["batch_size"] * 4,  # no gradients at evaluation, so larger batches fit
    num_train_epochs=cfg["epochs"],
    weight_decay=cfg["weight_decay"],
    warmup_steps=cfg["warmup"],  # a float below 1 is a share of the total steps
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",  # every epoch checkpoint is kept
    load_best_model_at_end=True,
    metric_for_best_model=cfg["best_metric"],
    greater_is_better=True,
    logging_steps=200,
    seed=cfg["seed"],
    report_to="none",
    remove_unused_columns=False,  # keeps soft_label and weight in the batches
)

trainer = SoftLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


# Phase 6. Train and save

start = time.time()
trainer.train()
minutes = (time.time() - start) / 60
print(f"Training time: {minutes:.1f} min")

trainer.save_model(run_dir / "final")
tokenizer.save_pretrained(run_dir / "final")

# Configuration actually used, with the training time
with open(metrics_dir / "config.yaml", "w") as f:
    yaml.safe_dump({**cfg, "training_minutes": round(minutes, 1)}, f, sort_keys=False)

# Full log (training loss every logging_steps, validation metrics every epoch) for the curves
log = pd.DataFrame(trainer.state.log_history)
log.to_csv(metrics_dir / "train_log.csv", index=False)

# One row per epoch with its validation metrics and checkpoint folder
epochs = log[log["eval_auc"].notna()].copy() if "eval_auc" in log else pd.DataFrame()
if not epochs.empty:
    epochs["checkpoint"] = "checkpoint-" + epochs["step"].astype(int).astype(str)
    keep = ["epoch", "checkpoint", "eval_loss", "eval_auc", "eval_pr_auc", "eval_f1", "eval_recall", "eval_precision"]
    epochs[keep].to_csv(metrics_dir / "epochs.csv", index=False)
    print(epochs[keep].to_string(index=False))
print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

# Validation probabilities of the best epoch, for ROC, PR and confusion matrices
output = trainer.predict(tokenized["validation"])
pd.DataFrame({
    "id": dataset["validation"]["id"],
    "label": output.label_ids,
    "prob_toxic": softmax(output.predictions, axis=1)[:, 1],
}).to_parquet(predictions_dir / "validation.parquet", index=False)


# Phase 7. Try the model on a few comments

classifier = pipeline("text-classification", model=str(run_dir / "final"), device=0)
for text in ["You are not an idiot", "You are an idiot", "I am a proud gay man"]:
    print(text, "->", classifier(text))
