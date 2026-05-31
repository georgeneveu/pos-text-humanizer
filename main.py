# =========================
# INSTALLATION (run once)
# =========================
# pip install spacy transformers torch sentencepiece
# python -m spacy download en_core_web_sm


import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# LOAD NLP MODEL (spaCy)
# =========================
# spaCy is used for Part-of-Speech (POS) tagging:
# - identifies verbs, adjectives, nouns, etc.
# - used to calculate "writing style metrics"

nlp = spacy.load("en_core_web_sm")


# =========================
# LOAD PARAPHRASING MODEL (T5)
# =========================
# This is a public HuggingFace model (NOT a key-based API)
# It rewrites sentences to different wording while keeping meaning

model_name = "Vamsi/T5_Paraphrase_Paws"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# =========================
# TARGET STYLE RULES
# =========================
# These are heuristic thresholds defining "human-like writing"

TARGET_VERB_RATIO = (0.10, 0.14)
TARGET_ADJ_RATIO = (0.10, 0.13)
TARGET_ADJ_VERB_RATIO = (0.8, 1.2)


# =========================
# TEXT ANALYSIS FUNCTION
# =========================
def analyze_metrics(text):
    """
    Uses spaCy POS tagging to measure sentence structure.

    Computes:
    - verb ratio
    - adjective ratio
    - adjective/verb ratio
    """

    doc = nlp(text)

    # Count verbs in text
    num_verbs = len([token for token in doc if token.pos_ == "VERB"])

    # Count adjectives in text
    num_adjs = len([token for token in doc if token.pos_ == "ADJ"])

    # Count only alphabetic tokens (ignore punctuation/numbers)
    num_tokens = len([token for token in doc if token.is_alpha])

    # Avoid division by zero
    verb_ratio = num_verbs / num_tokens if num_tokens else 0
    adj_ratio = num_adjs / num_tokens if num_tokens else 0
    adj_verb_ratio = num_adjs / max(1, num_verbs)

    return {
        "verbs": verb_ratio,
        "adjs": adj_ratio,
        "adj_verb_ratio": adj_verb_ratio
    }


# =========================
# PARAPHRASING FUNCTION (T5)
# =========================
def t5_paraphrase(sentence, max_length=128):
    """
    Rewrites a sentence using a transformer model (T5).

    No external API keys are used — model runs locally.
    """

    input_text = "paraphrase: " + sentence

    # Convert text to tokens
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate paraphrased output
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )

    # Convert tokens back to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# MAIN TEXT BALANCER ENGINE
# =========================
def balance_text_natural(text, max_iterations=2):
    """
    Main pipeline:
    1. Split text into sentences
    2. Analyze POS metrics
    3. Detect "unbalanced" sentences
    4. Rewrite them using T5
    5. Accept only if they fit target ranges
    """

    doc = nlp(text)

    # Sentence segmentation
    sentences = [sent.text.strip() for sent in doc.sents]

    balanced_sentences = []

    for sent in sentences:

        # Analyze current sentence structure
        metrics = analyze_metrics(sent)

        # Check if sentence is outside target ranges
        if (
            metrics["adj_verb_ratio"] > TARGET_ADJ_VERB_RATIO[1] or
            metrics["verbs"] < TARGET_VERB_RATIO[0] or
            metrics["adjs"] < TARGET_ADJ_RATIO[0]
        ):

            # Try rewriting multiple times
            for _ in range(max_iterations):

                new_sent = t5_paraphrase(sent)
                new_metrics = analyze_metrics(new_sent)

                # Accept only if it fits linguistic rules
                if (
                    TARGET_VERB_RATIO[0] <= new_metrics["verbs"] <= TARGET_VERB_RATIO[1] and
                    TARGET_ADJ_RATIO[0] <= new_metrics["adjs"] <= TARGET_ADJ_RATIO[1] and
                    TARGET_ADJ_VERB_RATIO[0] <= new_metrics["adj_verb_ratio"] <= TARGET_ADJ_VERB_RATIO[1]
                ):
                    sent = new_sent
                    break

        balanced_sentences.append(sent)

    # Rebuild full text
    balanced_text = " ".join(balanced_sentences)

    # Final metrics after processing
    overall_metrics = analyze_metrics(balanced_text)

    return balanced_text, overall_metrics


# =========================
# EXAMPLE RUN
# =========================
sample_text = """
Caterpillars are symbolic creatures representing transformation and hidden growth.
"""

print("Original metrics:", analyze_metrics(sample_text))

balanced_text, new_metrics = balance_text_natural(sample_text)

print("\nBalanced metrics:", new_metrics)
print("Balanced text:", balanced_text)
