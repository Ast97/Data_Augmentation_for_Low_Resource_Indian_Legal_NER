"""Preprocessing."""
import argparse
import io
from collections import Counter
from pathlib import Path

UNK = "<unk>"


def normalize_tok(tok):
    if tok.isnumeric():
        tok = "N"
    return tok


def _linearize(fout, train_file, vocab):
    """Convert column to one-per-line format."""
    with io.open(train_file, encoding="utf-8", errors="ignore") as fin:
        res = []
        for line in fin:
            line = line.strip()
            if not line:
                fout.write(" ".join(res))
                fout.write("\n")
                res = []
                continue
            cols = line.split("\t")
            if len(cols) != 2:
                tok, tag = cols[0], "O"  # Hardcode fix!
            else:
                tok, tag = cols
            tok = normalize_tok(tok)
            if tok not in vocab:
                tok = UNK
            if tag == "O":
                res += [tok]
            else:
                res += [tag,tok]


def linearize(train_file, vocab):
    """Convert column to one-per-line format."""
    out_file = Path(Path(train_file)).with_suffix(".lin.txt")
    print(f"Save to {out_file}")
    with io.open(out_file, "w", encoding="utf-8", errors="ignore") as fout:
        _linearize(fout, train_file, vocab)


def build_vocab(train_file, vocab_size, min_freq=1):
    """Build vocab from training set."""
    vocab = []
    with io.open(train_file, encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 2:
                continue
            vocab.append(normalize_tok(cols[0]))

    vocab = Counter(vocab)
    vocab_to_keep = set()
    with io.open("vocab.txt", "w", encoding="utf-8", errors="ignore") as fout:
        for tok, freq in vocab.most_common(vocab_size):
            if freq > min_freq:
                fout.write(f"{tok}\t{freq}\n")
                vocab_to_keep.add(tok)
    return vocab_to_keep


def build_args(parser):
    """Build arguments."""
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    return parser.parse_args()


def main():
    """Main workflow."""
    args = build_args(argparse.ArgumentParser())
    vocab = build_vocab(args.train_file, args.vocab_size)
    linearize(args.train_file, vocab)
    linearize(args.dev_file, vocab)


if __name__ == "__main__":
    main()