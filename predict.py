import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.dataset.dataset import encode_words_for_dependency
from src.dataset.char_vocab import encode_word_chars
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

    # Backward-compat: old checkpoints used a single hidden_dim
    arc_hidden_dim = train_args.get("arc_hidden_dim", train_args.get("hidden_dim", 512))
    label_hidden_dim = train_args.get("label_hidden_dim", train_args.get("hidden_dim", 512))

    upos2id = checkpoint.get("upos2id")
    char2id = checkpoint.get("char2id")

    model = DependencyParser(
        model_name=train_args["model_name"],
        arc_hidden_dim=arc_hidden_dim,
        label_hidden_dim=label_hidden_dim,
        n_labels=len(checkpoint["label2id"]),
        dropout=train_args["dropout"],
        n_pos=len(upos2id) if upos2id else 0,
        pos_dim=train_args.get("pos_dim", 0),
        n_chars=len(char2id) if char2id else 0,
        char_embed_dim=train_args.get("char_embed_dim", 50),
        char_out_dim=train_args.get("char_out_dim", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(train_args["model_name"], use_fast=True)
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}

    return model, tokenizer, id2label, train_args["max_len"], upos2id, char2id, train_args


def encode_sentence(tokens, tokenizer, max_len, upos_tags=None, upos2id=None,
                    char2id=None, max_word_len=30):
    encoding = encode_words_for_dependency(tokenizer, tokens, max_len)
    kept_tokens = encoding["kept_words"]
    kept_len = len(kept_tokens)

    features = {
        "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long),
        "word_starts": torch.tensor([encoding["word_starts"]], dtype=torch.long),
        "word_mask": torch.ones(1, kept_len, dtype=torch.bool),
    }

    if upos2id is not None:
        unk_idx = upos2id.get("<UNK>", 1)
        if upos_tags is not None:
            tags = upos_tags[:kept_len]
        else:
            tags = ["<UNK>"] * kept_len
        upos_ids = [upos2id.get(t, unk_idx) for t in tags]
        features["upos_ids"] = torch.tensor([upos_ids], dtype=torch.long)

    if char2id is not None:
        char_ids = [encode_word_chars(w, char2id, max_word_len) for w in kept_tokens]
        features["char_ids"] = torch.tensor([char_ids], dtype=torch.long)

    return features, kept_tokens


@torch.no_grad()
def predict_from_tokens(model, tokenizer, id2label, max_len, tokens, device,
                        upos_tags=None, upos2id=None, char2id=None, max_word_len=30):
    if not tokens:
        return ""
    batch, kept_tokens = encode_sentence(
        tokens, tokenizer, max_len,
        upos_tags=upos_tags, upos2id=upos2id,
        char2id=char2id, max_word_len=max_word_len,
    )
    batch = {key: value.to(device) for key, value in batch.items()}

    arc_scores, rel_scores, _ = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["word_starts"],
        batch["word_mask"],
        upos_ids=batch.get("upos_ids"),
        char_ids=batch.get("char_ids"),
    )
    # Always use MST at inference for valid parse trees (Phase 1)
    pred_heads, pred_labels = decode_predictions(
        arc_scores, rel_scores,
        word_mask=batch["word_mask"],
        use_mst=True,
    )

    pred_heads = pred_heads[0].tolist()
    pred_labels = pred_labels[0].tolist()

    lines = []
    for idx, token in enumerate(kept_tokens, start=1):
        head = pred_heads[idx - 1]
        label = id2label[pred_labels[idx - 1]]
        lines.append(f"{idx}\t{token}\t_\t_\t_\t_\t{head}\t{label}\t_\t_")

    return "\n".join(lines)


@torch.no_grad()
def predict_sentence(model, tokenizer, id2label, max_len, text, device,
                     upos2id=None, char2id=None, max_word_len=30):
    tokens = text.strip().split()
    return predict_from_tokens(
        model, tokenizer, id2label, max_len, tokens, device,
        upos2id=upos2id, char2id=char2id, max_word_len=max_word_len,
    )


def read_conllu_sentences(path):
    """Read CoNLL-U file returning list of (tokens, upos_tags) pairs."""
    sentences = []
    current_tokens = []
    current_upos = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current_tokens:
                    sentences.append((current_tokens, current_upos))
                    current_tokens = []
                    current_upos = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) == 10 and "-" not in cols[0] and "." not in cols[0]:
                current_tokens.append(cols[1])
                current_upos.append(cols[3] if cols[3] != "_" else "<UNK>")

    if current_tokens:
        sentences.append((current_tokens, current_upos))

    return sentences


def read_inputs(args):
    if args.text:
        return [(args.text.strip().split(), None)]  # (tokens, upos_tags)
    if args.input_file:
        if args.input_file.suffix == ".conllu":
            return read_conllu_sentences(args.input_file)
        # Plain text: one sentence per line
        with args.input_file.open("r", encoding="utf-8") as f:
            return [(line.strip().split(), None) for line in f if line.strip()]
    raise ValueError("You must provide either --text or --input-file.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, id2label, max_len, upos2id, char2id, train_args = load_model(
        args.checkpoint, device
    )
    max_word_len = train_args.get("max_word_len", 30)

    sentences = read_inputs(args)

    outputs = []
    for tokens, upos_tags in sentences:
        output = predict_from_tokens(
            model, tokenizer, id2label, max_len, tokens, device,
            upos_tags=upos_tags, upos2id=upos2id,
            char2id=char2id, max_word_len=max_word_len,
        )
        if output:
            outputs.append(output)

    content = "\n\n".join(outputs)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
