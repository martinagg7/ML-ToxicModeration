"""Gradio demo: compare the blocklist, TF-IDF + logistic regression and DistilBERT on one comment.

Run locally from the repository root:
    python app/app.py

The TF-IDF and DistilBERT models are loaded from ../models when they exist (local run),
otherwise they are downloaded from the Hugging Face Hub (Space).
"""

import re
from pathlib import Path

import gradio as gr
import joblib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
LOCAL_MODELS = APP_DIR.parent / "models"
BERT_REPO = "Martinagg/toxic-distilbert"
TFIDF_REPO = "Martinagg/toxic-tfidf-lr"

THRESHOLD = 0.5
REVIEW_BAND = (0.4, 0.6)  # abstention rule shown below the results

EXAMPLES = [
    "You are an idiot",
    "You are not an idiot",
    "I don't hate gay people",
    "Black people are not criminals",
    "I am a proud gay man",
    "My neighbor is black",
    "I am a Muslim woman",
    "Same-sex marriage should be legal",
    "Calling someone an idiot is not okay",
    "Saying gay people are disgusting is hateful",
    "This movie is fucking amazing",
    "I am an idiot and you are right",
]

# Blocklist: same regular expression as notebooks/03_blocklist.ipynb
with open(APP_DIR / "ldnoobw_en.txt") as f:
    entries = sorted({line.strip().lower() for line in f if line.strip()}, key=len, reverse=True)
pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in entries) + r")\b")

# TF-IDF + logistic regression
local_tfidf = LOCAL_MODELS / "tfidf_lr"
if local_tfidf.exists():
    vectorizer_path, classifier_path = local_tfidf / "vectorizer.joblib", local_tfidf / "classifier.joblib"
else:
    vectorizer_path = hf_hub_download(TFIDF_REPO, "vectorizer.joblib")
    classifier_path = hf_hub_download(TFIDF_REPO, "classifier.joblib")
vectorizer = joblib.load(vectorizer_path)
classifier = joblib.load(classifier_path)

# DistilBERT (soft labels + identity weights, epoch 2)
local_bert = LOCAL_MODELS / "soft_idw_lr2e-05_ep2" / "final"
bert_source = local_bert if local_bert.exists() else BERT_REPO
tokenizer = AutoTokenizer.from_pretrained(bert_source)
model = AutoModelForSequenceClassification.from_pretrained(bert_source).eval()


def decision(prob):
    return "Blocked" if prob >= THRESHOLD else "Published"


def moderate(comment):
    comment = comment.strip()
    if not comment:
        return [], ""

    matches = sorted(set(pattern.findall(comment.lower())))
    blocklist_row = ["Blocklist", "—", "Blocked" if matches else "Published",
                     ", ".join(matches) if matches else "no listed word"]

    prob_tfidf = float(classifier.predict_proba(vectorizer.transform([comment]))[0, 1])
    tfidf_row = ["TF-IDF + LR", f"{prob_tfidf:.2f}", decision(prob_tfidf), "word and bigram weights"]

    inputs = tokenizer(comment, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        prob_bert = float(torch.softmax(model(**inputs).logits, dim=1)[0, 1])
    bert_row = ["DistilBERT", f"{prob_bert:.2f}", decision(prob_bert), "reads the whole sentence in context"]

    note = (f"**Abstention rule:** comments with a probability between {REVIEW_BAND[0]} and {REVIEW_BAND[1]} "
            f"are sent to human review.")
    return [blocklist_row, tfidf_row, bert_row], note


with gr.Blocks(title="Trust in Automated Moderation") as demo:
    gr.Markdown(
        "# Trust in Automated Moderation of Toxic Comments\n"
        "Type a comment, or pick an example, and compare how three moderators classify it: "
        "a word **blocklist**, a **TF-IDF + logistic regression** classifier and a fine-tuned **DistilBERT**. "
        "Models trained on the Jigsaw Unintended Bias dataset (English news comments)."
    )
    comment = gr.Textbox(label="Comment", placeholder="Write a comment…", lines=2)
    button = gr.Button("Moderate", variant="primary")
    gr.Examples(EXAMPLES, inputs=comment, label="Probe sentences")
    results = gr.Dataframe(headers=["Model", "P(toxic)", "Decision", "How it decides"],
                           label="Results", interactive=False)
    note = gr.Markdown()

    button.click(moderate, inputs=comment, outputs=[results, note])
    comment.submit(moderate, inputs=comment, outputs=[results, note])

if __name__ == "__main__":
    demo.launch()
