from pathlib import Path

def _finalize_sentence(rows):
    if not rows:
        return None

    tokens = []
    upos = []
    heads = []
    labels = []

    for cols in rows:
        token_id = cols[0]
        if "-" in token_id or "." in token_id:
            continue

        tokens.append(cols[1])
        upos.append(cols[3])
        heads.append(int(cols[6]))
        labels.append(cols[7])

    if not tokens:
        return None

    return {
        "tokens": tokens,
        "upos": upos,
        "heads": heads,
        "labels": labels,
    }


def read_conllu(path):
    path = Path(path)
    sentences = []
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                sentence = _finalize_sentence(rows)
                if sentence is not None:
                    sentences.append(sentence)
                rows = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue
            rows.append(cols)

    sentence = _finalize_sentence(rows)
    if sentence is not None:
        sentences.append(sentence)

    return sentences


def build_label_vocab(sentences):
    labels = sorted({label for sentence in sentences for label in sentence["labels"]})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_upos_vocab(sentences):
    """Build upos2id / id2upos from training sentences.
    Index 0 is reserved for <PAD> (used for the virtual ROOT token).
    Index 1 is reserved for <UNK>.
    """
    tags = sorted({tag for sentence in sentences for tag in sentence["upos"]})
    special = ["<PAD>", "<UNK>"]
    all_tags = special + [t for t in tags if t not in special]
    upos2id = {tag: idx for idx, tag in enumerate(all_tags)}
    id2upos = {idx: tag for tag, idx in upos2id.items()}
    return upos2id, id2upos
