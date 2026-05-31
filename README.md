# POS Text Humanizer (part of speech tagging python) 
A Python NLP tool that analyzes and rewrites text using Part-of-Speech (POS) tagging and transformer-based paraphrasing to produce more natural, human-like writing. It evaluates verb and adjective ratios to detect unnatural patterns and rewrites text to improve grammatical balance and readability using iterative linguistic heuristics.

## 🧠 Overview

This project detects unnatural writing patterns by analyzing grammatical structure and then rewrites sentences to improve balance between verbs and adjectives.

It combines:
- spaCy for linguistic analysis (POS tagging)
- HuggingFace Transformers (T5 model) for paraphrasing
- Rule-based heuristics for style balancing

## ⚙️ How It Works

1. Input text is split into sentences
2. Each sentence is analyzed using POS tagging
3. Metrics are calculated:
   - Verb ratio
   - Adjective ratio
   - Adjective-to-verb ratio
4. Sentences that fall outside target ranges are flagged
5. A T5 model rewrites flagged sentences
6. The system iterates until the text meets target thresholds

## 📊 Target Linguistic Rules
AI-generated content has a distinctive fingerprint: an excess of adjectives and a lack of verbs. Human text features higher perplexity (57.3 vs 37.8) and burstiness (0.61 vs 0.38).
That’s why we’re rewriting the content to strip away predictable AI patterns and restore a natural, human flow.

| Metric | Target Range |
|--------|-------------|
| Verb Ratio | 10% – 14% |
| Adjective Ratio | 10% – 13% |
| Adj/Verb Ratio | 0.8 – 1.2 |

## 🚀 Installation

```bash
git clone https://github.com/georgeneveu/pos-text-humanizer.git
cd pos-text-humanizer

pip install -r requirements.txt
python -m spacy download en_core_web_sm
