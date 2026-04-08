import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dependency parsing predictions.")
    parser.add_argument("--gold-path", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument(
        "--truncate-pred-to-gold",
        action="store_true",
        help="If prediction has extra trailing sentences, keep only the prefix matching gold length.",
    )
    return parser.parse_args()


def read_conllu_annotations(path):
    sentences = []
    current = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue

            current.append(
                {
                    "id": cols[0],
                    "form": cols[1],
                    "head": cols[6],
                    "deprel": cols[7],
                }
            )

    if current:
        sentences.append(current)

    return sentences


def validate_alignment(gold_sentences, pred_sentences, truncate_pred_to_gold=False):
    if truncate_pred_to_gold and len(pred_sentences) > len(gold_sentences):
        pred_sentences = pred_sentences[: len(gold_sentences)]

    if len(gold_sentences) != len(pred_sentences):
        raise ValueError(
            f"Sentence count mismatch: gold={len(gold_sentences)}, pred={len(pred_sentences)}"
        )

    for sent_idx, (gold_sent, pred_sent) in enumerate(zip(gold_sentences, pred_sentences), start=1):
        if len(gold_sent) != len(pred_sent):
            raise ValueError(
                f"Token count mismatch at sentence {sent_idx}: "
                f"gold={len(gold_sent)}, pred={len(pred_sent)}"
            )
        for tok_idx, (gold_tok, pred_tok) in enumerate(zip(gold_sent, pred_sent), start=1):
            if gold_tok["form"] != pred_tok["form"]:
                raise ValueError(
                    f"Token mismatch at sentence {sent_idx}, token {tok_idx}: "
                    f"gold={gold_tok['form']!r}, pred={pred_tok['form']!r}"
                )

    return pred_sentences


def compute_scores(gold_sentences, pred_sentences):
    total = 0
    correct_heads = 0
    correct_labels = 0

    for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
        for gold_tok, pred_tok in zip(gold_sent, pred_sent):
            total += 1
            head_correct = gold_tok["head"] == pred_tok["head"]
            label_correct = gold_tok["deprel"] == pred_tok["deprel"]
            if head_correct:
                correct_heads += 1
            if head_correct and label_correct:
                correct_labels += 1

    if total == 0:
        return {"tokens": 0, "uas": 0.0, "las": 0.0}

    return {
        "tokens": total,
        "uas": correct_heads / total,
        "las": correct_labels / total,
    }


def main():
    args = parse_args()
    if not args.gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {args.gold_path}")
    if not args.pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")

    gold_sentences = read_conllu_annotations(args.gold_path)
    pred_sentences = read_conllu_annotations(args.pred_path)
    pred_sentences = validate_alignment(
        gold_sentences,
        pred_sentences,
        truncate_pred_to_gold=args.truncate_pred_to_gold,
    )
    scores = compute_scores(gold_sentences, pred_sentences)

    print(f"Tokens: {scores['tokens']}")
    print(f"UAS: {scores['uas']:.4f}")
    print(f"LAS: {scores['las']:.4f}")


if __name__ == "__main__":
    main()
