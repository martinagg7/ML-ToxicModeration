"""Predict the probability of toxicity with a trained DistilBERT checkpoint.

Usage:
    python src/predict_distilbert.py --model models/soft_idw_lr2e-05_ep2/final --split test \
        --out results/predictions/soft_idw_lr2e-05_ep2/test.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/jigsaw_clean.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--split", default="test")
parser.add_argument("--out", required=True)
parser.add_argument("--max_length", type=int, default=128)
args = parser.parse_args()

df = pd.read_parquet(DATA_PATH, columns=["id", "comment_text", "label", "split"])
df = df[df["split"] == args.split].drop(columns="split")

tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model)
model = AutoModelForSequenceClassification.from_pretrained(ROOT / args.model)

dataset = Dataset.from_pandas(df, preserve_index=False).map(
    lambda e: tokenizer(e["comment_text"], truncation=True, max_length=args.max_length),
    batched=True, num_proc=4, remove_columns=["comment_text"],
)

training_args = TrainingArguments(output_dir="/tmp/predict_distilbert", per_device_eval_batch_size=64,
                                  bf16=True, report_to="none")
output = Trainer(model=model, args=training_args, processing_class=tokenizer).predict(dataset)

out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "id": df["id"].values,
    "label": df["label"].values,
    "prob_toxic": softmax(output.predictions, axis=1)[:, 1],
}).to_parquet(out, index=False)
print(f"Saved {len(df):,} predictions to {out}")
