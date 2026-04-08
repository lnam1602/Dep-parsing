def build_char_vocab(sentences):
    """Build char2id / id2char from training sentences.
    Index 0 is reserved for <PAD>, index 1 for <UNK>.
    """
    chars = sorted({ch for sent in sentences for word in sent["tokens"] for ch in word})
    special = ["<PAD>", "<UNK>"]
    all_chars = special + [c for c in chars if c not in special]
    char2id = {c: i for i, c in enumerate(all_chars)}
    id2char = {i: c for c, i in char2id.items()}
    return char2id, id2char


def encode_word_chars(word, char2id, max_word_len=30, pad_idx=0, unk_idx=1):
    """Encode a word as a padded/truncated list of character ids."""
    ids = [char2id.get(ch, unk_idx) for ch in word[:max_word_len]]
    ids += [pad_idx] * (max_word_len - len(ids))
    return ids
