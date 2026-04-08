import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VLSP 2020 dependency parsing data.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/DP-2020/TrainingData"),
        help="Directory containing gold training files.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/DP-2020/DataTestGoldDP2020"),
        help="Directory containing gold test annotation files (*_gold.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_data/dp2020"),
        help="Directory for generated train/dev/test files.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Fraction of training sentences reserved for dev.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/dev split.",
    )
    return parser.parse_args()


def read_sentences(path):
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
            current.append(line)

    if current:
        sentences.append(current)

    return sentences


def write_conllu(path, sentences):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + "\n")
            f.write("\n")


def strip_labels(sentences):
    """Replace HEAD (col 6) and DEPREL (col 7) with '_' to create input-only CoNLL-U."""
    result = []
    for sentence in sentences:
        stripped = []
        for line in sentence:
            if line.startswith("#"):
                stripped.append(line)
                continue
            cols = line.split("\t")
            if len(cols) == 10:
                cols[6] = "_"
                cols[7] = "_"
                line = "\t".join(cols)
            stripped.append(line)
        result.append(stripped)
    return result


def collect_sentences(files):
    all_sentences = []
    for file_path in sorted(files):
        sentences = read_sentences(file_path)
        for sentence in sentences:
            all_sentences.append([f"# source_file = {file_path.name}"] + sentence)
    return all_sentences


def main():
    args = parse_args()

    train_files = sorted(args.train_dir.glob("*.txt"))
    gold_files = sorted(f for f in args.gold_dir.glob("*_gold.txt")
                        if "total" not in f.name)

    if not train_files:
        raise FileNotFoundError(f"No training files found in {args.train_dir}")

    gold_sentences = collect_sentences(train_files)

    rng = random.Random(args.seed)
    rng.shuffle(gold_sentences)

    dev_size = max(1, int(len(gold_sentences) * args.dev_ratio))
    dev_sentences = gold_sentences[:dev_size]
    train_sentences = gold_sentences[dev_size:]

    if not train_sentences:
        raise ValueError("Train split is empty. Reduce --dev-ratio.")

    train_path = args.output_dir / "train.conllu"
    dev_path = args.output_dir / "dev.conllu"

    write_conllu(train_path, train_sentences)
    write_conllu(dev_path, dev_sentences)

    # Tạo total-gold.conllu và test_gold_input.conllu từ gold files
    if gold_files:
        gold_sentences = collect_sentences(gold_files)
        total_gold_path = args.output_dir / "total-gold.conllu"
        test_gold_input_path = args.output_dir / "test_gold_input.conllu"
        write_conllu(total_gold_path, gold_sentences)
        write_conllu(test_gold_input_path, strip_labels(gold_sentences))
        print(f"Saved total gold ({len(gold_sentences)} sentences) to {total_gold_path}")
        print(f"Saved test gold input to {test_gold_input_path}")
    else:
        print(f"Warning: no *_gold.txt files found in {args.gold_dir}")

    summary_path = args.output_dir / "README.txt"
    n_gold = len(gold_sentences) if gold_files else 0
    summary = [
        "Generated from source directories:",
        f"- train_dir: {args.train_dir}",
        f"- gold_dir:  {args.gold_dir}",
        "",
        "This directory is derived data and does not modify source files.",
        "",
        "Split statistics:",
        f"- train sentences:      {len(train_sentences)}",
        f"- dev sentences:        {len(dev_sentences)}",
        f"- test gold sentences:  {n_gold}",
        f"- seed:                 {args.seed}",
        f"- dev_ratio:            {args.dev_ratio}",
        "",
        "Files:",
        f"- {train_path.name}          : training data (gold labels)",
        f"- {dev_path.name}            : development data (gold labels)",
        f"- total-gold.conllu          : test gold annotations ({n_gold} sentences)",
        f"- test_gold_input.conllu     : test input without labels — use for predict.py",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved train split to {train_path}")
    print(f"Saved dev split to {dev_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
