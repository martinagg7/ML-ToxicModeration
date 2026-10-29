"""Fine-tune DistilBERT for toxicity classification on the Jigsaw split.

Usage:
    python src/train_distilbert.py                          # values from configs/base.yaml
    python src/train_distilbert.py --train_size 20000       # override any value
    python src/train_distilbert.py --freeze_encoder true --learning_rate 1e-3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


# Configuration: base.yaml, then command line overrides

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/base.yaml")
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

    mode = "frozen" if cfg["freeze_encoder"] else "full"
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

# Frozen encoder: only the classification head (pre_classifier + classifier) is trained
if cfg["freeze_encoder"]:
    for param in model.base_model.parameters():
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,} of {model.num_parameters():,}")


# Phase 2. Build the DatasetDict from the clean split

df = pd.read_parquet(DATA_PATH, columns=["id", "comment_text", "label", "split"])


def to_dataset(name):
    return Dataset.from_pandas(df[df["split"] == name].drop(columns="split"), preserve_index=False)


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


# Phase 3. Tokenize

def preprocess_function(examples):
    return tokenizer(examples["comment_text"], truncation=True, max_length=cfg["max_length"])


# The test set is not tokenized here: it is only used in the final evaluation
tokenized = DatasetDict({
    name: dataset[name].map(preprocess_function, batched=True, num_proc=4)
    for name in ["train", "validation"]
})


# Phase 4. Evaluation metrics

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


# Phase 5. Training settings and Trainer

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
)

trainer = Trainer(
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

# Validation probabilities of the best model, for ROC, PR and confusion matrices
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
