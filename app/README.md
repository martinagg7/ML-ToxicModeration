---
title: Trust in Automated Moderation
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# Trust in Automated Moderation of Toxic Comments

Interactive demo for the ECE 4424 / CS 4824 capstone project by Martina García González.

Type a comment and compare how three moderators classify it:

- **Blocklist**: blocks a comment if it contains a word from the [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) English list.
- **TF-IDF + logistic regression**: word unigrams and bigrams, trained on Jigsaw Unintended Bias.
- **DistilBERT**: fine-tuned with soft labels and identity weights on Jigsaw Unintended Bias.

A comment is blocked when P(toxic) ≥ 0.5; probabilities between 0.4 and 0.6 are flagged for human review.

Code and report: https://github.com/martinagg7/ML-ToxicModeration
