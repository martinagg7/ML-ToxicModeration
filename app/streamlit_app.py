"""Streamlit demo: compare the blocklist, TF-IDF + logistic regression and DistilBERT on one comment.

Run locally from the repository root:
    streamlit run app/streamlit_app.py

The TF-IDF and DistilBERT models are loaded from ../models when they exist (local run),
otherwise they are downloaded from the Hugging Face Hub (Streamlit Community Cloud).
"""

import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
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


@st.cache_resource
def load_models():
    # Blocklist: same regular expression as notebooks/03_blocklist.ipynb
    with open(APP_DIR / "ldnoobw_en.txt") as f:
        entries = sorted({line.strip().lower() for line in f if line.strip()}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in entries) + r")\b")

    local_tfidf = LOCAL_MODELS / "tfidf_lr"
    if local_tfidf.exists():
        vectorizer_path, classifier_path = local_tfidf / "vectorizer.joblib", local_tfidf / "classifier.joblib"
    else:
        vectorizer_path = hf_hub_download(TFIDF_REPO, "vectorizer.joblib")
        classifier_path = hf_hub_download(TFIDF_REPO, "classifier.joblib")

    local_bert = LOCAL_MODELS / "soft_idw_lr2e-05_ep2" / "final"
    bert_source = local_bert if local_bert.exists() else BERT_REPO
    tokenizer = AutoTokenizer.from_pretrained(bert_source)
    model = AutoModelForSequenceClassification.from_pretrained(bert_source).eval()
    return pattern, joblib.load(vectorizer_path), joblib.load(classifier_path), tokenizer, model


def decision(prob):
    return "Blocked" if prob >= THRESHOLD else "Published"


def moderate(comment):
    pattern, vectorizer, classifier, tokenizer, model = load_models()

    matches = sorted(set(pattern.findall(comment.lower())))
    blocklist_row = ["Blocklist", "—", "Blocked" if matches else "Published",
                     ", ".join(matches) if matches else "no listed word"]

    prob_tfidf = float(classifier.predict_proba(vectorizer.transform([comment]))[0, 1])
    tfidf_row = ["TF-IDF + LR", f"{prob_tfidf:.2f}", decision(prob_tfidf), "word and bigram weights"]

    inputs = tokenizer(comment, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        prob_bert = float(torch.softmax(model(**inputs).logits, dim=1)[0, 1])
    bert_row = ["DistilBERT", f"{prob_bert:.2f}", decision(prob_bert), "reads the whole sentence in context"]

    return pd.DataFrame([blocklist_row, tfidf_row, bert_row],
                        columns=["Model", "P(toxic)", "Decision", "How it decides"])


st.set_page_config(page_title="Trust in Automated Moderation", layout="centered")
st.title("Trust in Automated Moderation of Toxic Comments")
st.markdown(
    "Type a comment, or pick an example, and compare how three moderators classify it: "
    "a word **blocklist**, a **TF-IDF + logistic regression** classifier and a fine-tuned **DistilBERT**. "
    "Models trained on the Jigsaw Unintended Bias dataset (English news comments)."
)

if "comment" not in st.session_state:
    st.session_state.comment = ""

st.caption("Probe sentences")
columns = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if columns[i % 3].button(example, width="stretch"):
        st.session_state.comment = example

comment = st.text_area("Comment", key="comment", placeholder="Write a comment…", height=80)

if st.button("Moderate", type="primary") or comment.strip():
    if comment.strip():
        with st.spinner("Loading models…"):
            results = moderate(comment.strip())
        st.dataframe(results, hide_index=True, width="stretch")
        st.markdown(f"**Abstention rule:** comments with a probability between {REVIEW_BAND[0]} and "
                    f"{REVIEW_BAND[1]} are sent to human review.")

st.caption("Code and report: [github.com/martinagg7/ML-ToxicModeration](https://github.com/martinagg7/ML-ToxicModeration) · "
           "Models: [Martinagg/toxic-distilbert](https://huggingface.co/Martinagg/toxic-distilbert), "
           "[Martinagg/toxic-tfidf-lr](https://huggingface.co/Martinagg/toxic-tfidf-lr)")
