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
        "--test-dir",
        type=Path,
        default=Path("data/DP-2020/DataTestDP2020-CoNLLU"),
        help="Directory containing test input files in CoNLL-U-like format.",
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
    test_files = sorted(args.test_dir.glob("*.conllu"))

    if not train_files:
        raise FileNotFoundError(f"No training files found in {args.train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No test files found in {args.test_dir}")

    gold_sentences = collect_sentences(train_files)
    test_sentences = collect_sentences(test_files)

    rng = random.Random(args.seed)
    rng.shuffle(gold_sentences)

    dev_size = max(1, int(len(gold_sentences) * args.dev_ratio))
    dev_sentences = gold_sentences[:dev_size]
    train_sentences = gold_sentences[dev_size:]

    if not train_sentences:
        raise ValueError("Train split is empty. Reduce --dev-ratio.")

    train_path = args.output_dir / "train.conllu"
    dev_path = args.output_dir / "dev.conllu"
    test_path = args.output_dir / "test_input.conllu"

    write_conllu(train_path, train_sentences)
    write_conllu(dev_path, dev_sentences)
    write_conllu(test_path, test_sentences)

    summary_path = args.output_dir / "README.txt"
    summary = [
        "Generated from source directories:",
        f"- train_dir: {args.train_dir}",
        f"- test_dir: {args.test_dir}",
        "",
        "This directory is derived data and does not modify source files.",
        f"- train sentences: {len(train_sentences)}",
        f"- dev sentences: {len(dev_sentences)}",
        f"- test sentences: {len(test_sentences)}",
        f"- seed: {args.seed}",
        f"- dev_ratio: {args.dev_ratio}",
        "",
        "Files:",
        f"- {train_path.name}",
        f"- {dev_path.name}",
        f"- {test_path.name}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved train split to {train_path}")
    print(f"Saved dev split to {dev_path}")
    print(f"Saved test input to {test_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
