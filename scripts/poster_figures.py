"""Figures for the milestone poster, computed from the saved test predictions.

Usage:
    python scripts/poster_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports/poster"

MAIN_IDENTITIES = {
    "male": "Male", "female": "Female", "homosexual_gay_or_lesbian": "Gay / lesbian",
    "christian": "Christian", "jewish": "Jewish", "muslim": "Muslim",
    "black": "Black", "white": "White", "psychiatric_or_mental_illness": "Mental illness",
}
MODELS = {"Blocklist": "#9AA3AB", "TF-IDF + LR": "#2E6B4F", "DistilBERT": "#123B5C"}  # poster palette

identities = pd.read_parquet(ROOT / "data/processed/jigsaw_clean.parquet",
                             columns=["id", "split"] + list(MAIN_IDENTITIES))
identities = identities[identities["split"] == "test"].drop(columns="split")

blocklist = pd.read_parquet(ROOT / "results/predictions/blocklist/test.parquet")
tfidf = pd.read_parquet(ROOT / "results/predictions/tfidf/test.parquet")
bert = pd.read_parquet(ROOT / "results/predictions/soft_idw_lr2e-05_ep2/test.parquet")
preds = {
    "Blocklist": blocklist.assign(pred=blocklist["blocked"]),
    "TF-IDF + LR": tfidf,
    "DistilBERT": bert.assign(pred=(bert["prob_toxic"] >= 0.5).astype(int)),
}


def fpr(p, mask):
    non_toxic = p[mask & (p["label"] == 0)]
    return non_toxic["pred"].mean() * 100


rows = {}
for name, p in preds.items():
    p = identities.merge(p[["id", "label", "pred"]], on="id")
    values = {"All comments": fpr(p, pd.Series(True, index=p.index))}
    for col, label in MAIN_IDENTITIES.items():
        values[label] = fpr(p, p[col] >= 0.5)
    rows[name] = values
table = pd.DataFrame(rows)
print(table.round(1))

# False positive rate by identity, three models
groups = list(table.index)
x = np.arange(len(groups))
width = 0.27

fig, ax = plt.subplots(figsize=(11.6, 3.6))
for k, (name, color) in enumerate(MODELS.items()):
    bars = ax.bar(x + (k - 1) * width, table[name], width, color=color, label=name)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in table[name]], padding=2, fontsize=8.5)

ax.axvline(0.5, color="#9aa5b1", linewidth=1)
ax.set_xticks(x, groups, fontsize=10.5)
ax.set_ylabel("Non-toxic comments\nwrongly blocked (%)", fontsize=10.5)
ax.set_ylim(0, table.max().max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
plt.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fpr_by_identity.png", dpi=300)
print(f"Saved {OUT_DIR / 'fpr_by_identity.png'}")

# Same figure as horizontal bars, for a narrow poster column
y = np.arange(len(groups))
height = 0.27

fig, ax = plt.subplots(figsize=(5.8, 3.6))
for k, (name, color) in enumerate(MODELS.items()):
    bars = ax.barh(y + (k - 1) * height, table[name], height, color=color, label=name)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in table[name]], padding=2, fontsize=6.5)
ax.set_ylim(len(groups) - 0.45, -0.55)  # top to bottom, no extra margin
ax.set_yticks(y, groups, fontsize=8.5)
ax.set_xlabel("Non-toxic comments wrongly blocked (%)", fontsize=8.5)
ax.set_xlim(0, table.max().max() * 1.12)
ax.tick_params(axis="x", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
plt.tight_layout()
fig.savefig(OUT_DIR / "fpr_by_identity_column.png", dpi=300)
print(f"Saved {OUT_DIR / 'fpr_by_identity_column.png'}")

# Share of toxic comments by identity mentioned (whole cleaned dataset)
data = pd.read_parquet(ROOT / "data/processed/jigsaw_clean.parquet", columns=["label"] + list(MAIN_IDENTITIES))
toxic_rate = pd.Series({label: data.loc[data[col] >= 0.5, "label"].mean() * 100
                        for col, label in MAIN_IDENTITIES.items()}).sort_values()
overall = data["label"].mean() * 100

toxic_rate = toxic_rate.iloc[::-1]
fig, ax = plt.subplots(figsize=(5.7, 1.6))
bars = ax.bar(toxic_rate.index, toxic_rate.values, color="#123B5C", width=0.65)
ax.bar_label(bars, labels=[f"{v:.0f}%" for v in toxic_rate.values], padding=1, fontsize=7)
ax.axhline(overall, color="#C0392B", linestyle="--", linewidth=1, label=f"All comments ({overall:.0f}%)")
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_ylabel("Toxic (%)", fontsize=7.5)
ax.set_ylim(0, toxic_rate.max() * 1.2)
ax.tick_params(axis="x", labelsize=7, length=0)
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "toxic_by_identity.png", dpi=300)
print(f"Saved {OUT_DIR / 'toxic_by_identity.png'}")

# Precision-recall curves on the test set
from sklearn.metrics import average_precision_score, precision_recall_curve

scores = {"TF-IDF + LR": tfidf["prob_toxic"], "DistilBERT": bert["prob_toxic"]}
labels = {"TF-IDF + LR": tfidf["label"], "DistilBERT": bert["label"]}

fig, ax = plt.subplots(figsize=(2.8, 2.2))
for name in ["TF-IDF + LR", "DistilBERT"]:
    precision, recall, _ = precision_recall_curve(labels[name], scores[name])
    ap = average_precision_score(labels[name], scores[name])
    ax.plot(recall, precision, color=MODELS[name], linewidth=1.8, label=f"{name.replace(' + LR', '')}: {ap:.2f}")

# the blocklist has no probability, only a fixed rule: one operating point
blocked, y_block = blocklist["blocked"], blocklist["label"]
block_precision = y_block[blocked == 1].mean()
block_recall = blocked[y_block == 1].mean()
ax.scatter(block_recall, block_precision, color=MODELS["Blocklist"], edgecolor="#26323A", s=30, zorder=3)
ax.annotate("Blocklist", (block_recall, block_precision), xytext=(0, 7), textcoords="offset points", ha="center",
            fontsize=6.5, va="bottom")
ax.axhline(bert["label"].mean(), color="#9aa5b1", linestyle="--", linewidth=0.8)
ax.set_xlabel("Recall", fontsize=8)
ax.set_ylabel("Precision", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.3, 0.1), title="PR-AUC", title_fontsize=6.5, alignment="left")
plt.tight_layout()
fig.savefig(OUT_DIR / "pr_curves.png", dpi=300)
print(f"Saved {OUT_DIR / 'pr_curves.png'}")

# Compact versions for the two-column "The Evidence" box of the poster
target = pd.read_parquet(ROOT / "data/processed/jigsaw_clean.parquet", columns=["id", "target"])
BANDS = ["< 0.4", "0.4 - 0.6", "> 0.6"]

# False positive rate by identity, narrow and tall, no value labels
fig, ax = plt.subplots(figsize=(3.4, 3.3))
for k, (name, color) in enumerate(MODELS.items()):
    bars = ax.barh(y + (k - 1) * height, table[name], height, color=color, label=name)
    if name == "DistilBERT":  # label only the main model to keep the chart readable
        for yi, v in zip(y + (k - 1) * height, table[name]):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center_baseline", ha="left", fontsize=6, color=color, fontweight="bold")
ax.set_yticks(y, groups, fontsize=7.5)
ax.set_ylim(len(groups) - 0.45, -0.55)
ax.set_xlabel("Non-toxic comments wrongly blocked (%)", fontsize=7.5)
ax.tick_params(axis="x", labelsize=7)
ax.xaxis.grid(True, color="#e3e6ea", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(OUT_DIR / "fpr_by_identity_poster.png", dpi=300)
print(f"Saved {OUT_DIR / 'fpr_by_identity_poster.png'}")

# Precision-recall, small square version
fig, ax = plt.subplots(figsize=(2.1, 2.1))
for name in ["TF-IDF + LR", "DistilBERT"]:
    precision, recall, _ = precision_recall_curve(labels[name], scores[name])
    ap = average_precision_score(labels[name], scores[name])
    ax.plot(recall, precision, color=MODELS[name], linewidth=1.6, label=f"{name.replace(' + LR', '')}: {ap:.2f}")
ax.scatter(block_recall, block_precision, color=MODELS["Blocklist"], edgecolor="#26323A", s=22, zorder=3)
ax.annotate("Blocklist", (block_recall, block_precision), xytext=(0, 6), textcoords="offset points", ha="left",
            fontsize=6, va="bottom")
ax.set_xlabel("Recall", fontsize=7)
ax.set_ylabel("Precision", fontsize=7)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.tick_params(labelsize=6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.2, 0.03), title="PR-AUC", title_fontsize=5.5,
          alignment="left")
plt.tight_layout()
fig.savefig(OUT_DIR / "pr_curves_poster.png", dpi=300)
print(f"Saved {OUT_DIR / 'pr_curves_poster.png'}")

# Error rate by target band
banded = {}
for name, p in [("Blocklist", blocklist.assign(pred=blocklist["blocked"], score=np.nan)),
                ("TF-IDF + LR", tfidf.assign(score=tfidf["prob_toxic"])),
                ("DistilBERT", bert.assign(pred=(bert["prob_toxic"] >= 0.5).astype(int), score=bert["prob_toxic"]))]:
    p = p.merge(target, on="id")
    p["band"] = pd.cut(p["target"], [0, 0.4, 0.6, 1.01], labels=BANDS, right=False)
    banded[name] = p
error_rate = pd.DataFrame({name: p.groupby("band", observed=True).apply(
    lambda d: (d["pred"] != d["label"]).mean() * 100, include_groups=False) for name, p in banded.items()})
print(error_rate.round(1))

x = np.arange(len(BANDS))
width = 0.27
fig, ax = plt.subplots(figsize=(3.4, 1.55))
for k, (name, color) in enumerate(MODELS.items()):
    bars = ax.bar(x + (k - 1) * width, error_rate[name], width, color=color, label=name)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in error_rate[name]], padding=1, fontsize=6)
ax.set_xticks(x, ["< 0.4\nclearly non-toxic", "0.4 - 0.6\ndoubtful", "> 0.6\nclearly toxic"], fontsize=6.5)
ax.set_ylabel("Error rate (%)", fontsize=7)
ax.set_ylim(0, 100)
ax.tick_params(axis="y", labelsize=6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6, loc="upper left")
plt.tight_layout()
fig.savefig(OUT_DIR / "errors_by_band.png", dpi=300)
print(f"Saved {OUT_DIR / 'errors_by_band.png'}")

# Predicted probability by target band, both models in one axes
fig, ax = plt.subplots(figsize=(3.4, 1.7))
for k, name in enumerate(["TF-IDF + LR", "DistilBERT"]):
    p = banded[name]
    data = [p.loc[p["band"] == b, "score"] for b in BANDS]
    positions = np.arange(len(BANDS)) + (k - 0.5) * 0.36
    box = ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True, showfliers=False,
                     medianprops={"color": "#F28E2B", "linewidth": 1.6}, whiskerprops={"linewidth": 0.8},
                     capprops={"linewidth": 0.8})
    for patch in box["boxes"]:
        patch.set_facecolor(MODELS[name])
        patch.set_edgecolor(MODELS[name])
    ax.plot([], [], color=MODELS[name], linewidth=5, label=name)
ax.axhline(0.5, color="#9aa5b1", linestyle="--", linewidth=0.8)
ax.set_xticks(np.arange(len(BANDS)), ["< 0.4\nclearly non-toxic", "0.4 - 0.6\ndoubtful", "> 0.6\nclearly toxic"],
              fontsize=6.5)
ax.set_ylabel("P(toxic)", fontsize=7)
ax.set_ylim(-0.02, 1.02)
ax.tick_params(axis="y", labelsize=6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6, loc="upper left")
plt.tight_layout()
fig.savefig(OUT_DIR / "prob_by_band.png", dpi=300)
print(f"Saved {OUT_DIR / 'prob_by_band.png'}")
