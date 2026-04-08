import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def encode_words_for_dependency(tokenizer, words, max_len):
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_budget = max_len - special_tokens
    if token_budget <= 0:
        raise ValueError("max_len is too small for tokenizer special tokens.")

    subword_tokens = []
    word_starts = []
    kept_words = []

    for word in words:
        pieces = tokenizer.tokenize(word)
        if not pieces:
            pieces = [tokenizer.unk_token]

        if len(subword_tokens) + len(pieces) > token_budget:
            break

        word_starts.append(len(subword_tokens) + 1)
        subword_tokens.extend(pieces)
        kept_words.append(word)

    input_ids = tokenizer.build_inputs_with_special_tokens(
        tokenizer.convert_tokens_to_ids(subword_tokens)
    )
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_starts": word_starts,
        "kept_words": kept_words,
    }


class DependencyDataset(Dataset):
    def __init__(
        self,
        sentences,
        label2id,
        model_name="vinai/phobert-base",
        max_len=128,
        upos2id=None,
        char2id=None,
        max_word_len=30,
    ):
        self.sentences = sentences
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.upos2id = upos2id
        self.char2id = char2id
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]

        words = sent["tokens"]
        heads = sent["heads"]
        labels = sent["labels"]

        encoding = encode_words_for_dependency(self.tokenizer, words, self.max_len)
        kept_len = len(encoding["kept_words"])
        kept_words = encoding["kept_words"]
        heads = heads[:kept_len]
        label_ids = [self.label2id[label] for label in labels[:kept_len]]

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "word_starts": torch.tensor(encoding["word_starts"], dtype=torch.long),
            "heads": torch.tensor(heads, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "word_mask": torch.ones(kept_len, dtype=torch.bool),
        }

        if self.upos2id is not None:
            upos_tags = sent["upos"][:kept_len]
            unk_idx = self.upos2id.get("<UNK>", 1)
            upos_ids = [self.upos2id.get(tag, unk_idx) for tag in upos_tags]
            item["upos_ids"] = torch.tensor(upos_ids, dtype=torch.long)

        if self.char2id is not None:
            from .char_vocab import encode_word_chars
            char_ids = [
                encode_word_chars(w, self.char2id, self.max_word_len)
                for w in kept_words
            ]
            item["char_ids"] = torch.tensor(char_ids, dtype=torch.long)

        return item


def collate_dependency_batch(batch):
    batch_size = len(batch)
    max_subwords = max(item["input_ids"].size(0) for item in batch)
    max_words = max(item["word_starts"].size(0) for item in batch)

    input_ids = torch.zeros(batch_size, max_subwords, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_subwords, dtype=torch.long)
    word_starts = torch.full((batch_size, max_words), -1, dtype=torch.long)
    heads = torch.full((batch_size, max_words), -100, dtype=torch.long)
    labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)

    has_upos = "upos_ids" in batch[0]
    has_char = "char_ids" in batch[0]

    upos_ids_batch = None
    if has_upos:
        upos_ids_batch = torch.zeros(batch_size, max_words, dtype=torch.long)

    char_ids_batch = None
    if has_char:
        max_word_len = max(item["char_ids"].size(1) for item in batch)
        char_ids_batch = torch.zeros(batch_size, max_words, max_word_len, dtype=torch.long)

    for i, item in enumerate(batch):
        subword_len = item["input_ids"].size(0)
        word_len = item["word_starts"].size(0)

        input_ids[i, :subword_len] = item["input_ids"]
        attention_mask[i, :subword_len] = item["attention_mask"]
        word_starts[i, :word_len] = item["word_starts"]
        heads[i, :word_len] = item["heads"]
        labels[i, :word_len] = item["labels"]
        word_mask[i, :word_len] = item["word_mask"]

        if has_upos:
            upos_ids_batch[i, :word_len] = item["upos_ids"]

        if has_char:
            char_len = item["char_ids"].size(1)
            char_ids_batch[i, :word_len, :char_len] = item["char_ids"]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_starts": word_starts,
        "heads": heads,
        "labels": labels,
        "word_mask": word_mask,
    }
    if has_upos:
        result["upos_ids"] = upos_ids_batch
    if has_char:
        result["char_ids"] = char_ids_batch

    return result
