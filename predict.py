import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.dataset.dataset import encode_words_for_dependency
from src.model.parser import DependencyParser
from src.training.metrics import decode_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Predict dependency parses in CoNLL-U format.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train.py first or provide the correct --checkpoint path."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]

    model = DependencyParser(
        model_name=train_args["model_name"],
        hidden_dim=train_args["hidden_dim"],
        n_labels=len(checkpoint["label2id"]),
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(train_args["model_name"], use_fast=True)
    id2label = checkpoint["id2label"]
    id2label = {int(k): v for k, v in id2label.items()}

    return model, tokenizer, id2label, train_args["max_len"]


def encode_sentence(tokens, tokenizer, max_len):
    encoding = encode_words_for_dependency(tokenizer, tokens, max_len)
    kept_tokens = encoding["kept_words"]

    features = {
        "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long),
        "word_starts": torch.tensor([encoding["word_starts"]], dtype=torch.long),
        "word_mask": torch.ones(1, len(encoding["word_starts"]), dtype=torch.bool),
    }
    return features, kept_tokens


@torch.no_grad()
def predict_from_tokens(model, tokenizer, id2label, max_len, tokens, device):
    if not tokens:
        return ""
    batch, kept_tokens = encode_sentence(tokens, tokenizer, max_len)
    batch = {key: value.to(device) for key, value in batch.items()}

    arc_scores, rel_scores, _ = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["word_starts"],
        batch["word_mask"],
    )
    pred_heads, pred_labels = decode_predictions(arc_scores, rel_scores)

    pred_heads = pred_heads[0].tolist()
    pred_labels = pred_labels[0].tolist()

    lines = []
    for idx, token in enumerate(kept_tokens, start=1):
        head = pred_heads[idx - 1]
        label = id2label[pred_labels[idx - 1]]
        lines.append(f"{idx}\t{token}\t_\t_\t_\t_\t{head}\t{label}\t_\t_")

    return "\n".join(lines)


@torch.no_grad()
def predict_sentence(model, tokenizer, id2label, max_len, text, device):
    tokens = text.strip().split()
    return predict_from_tokens(model, tokenizer, id2label, max_len, tokens, device)


def read_conllu_sentences(path):
    sentences = []
    current_tokens = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current_tokens:
                    sentences.append(current_tokens)
                    current_tokens = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) == 10 and "-" not in cols[0] and "." not in cols[0]:
                current_tokens.append(cols[1])

    if current_tokens:
        sentences.append(current_tokens)

    return sentences


def read_inputs(args):
    if args.text:
        return [args.text]
    if args.input_file:
        if args.input_file.suffix == ".conllu":
            return read_conllu_sentences(args.input_file)
        with args.input_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("You must provide either --text or --input-file.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, id2label, max_len = load_model(args.checkpoint, device)
    sentences = read_inputs(args)

    outputs = []
    for sentence in sentences:
        if isinstance(sentence, list):
            outputs.append(predict_from_tokens(model, tokenizer, id2label, max_len, sentence, device))
        else:
            outputs.append(predict_sentence(model, tokenizer, id2label, max_len, sentence, device))

    content = "\n\n".join(output for output in outputs if output)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
