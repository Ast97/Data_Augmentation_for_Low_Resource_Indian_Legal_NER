"""Preprocessing."""
import argparse
import io
import json


def _write(toks, label, fout):
    for i, tok in enumerate(toks):
        if label == "O":
            fout.write("\t".join([tok, label]))
        elif i==0:
            fout.write("\t".join([tok, "B-" + label]))
        else:
            fout.write("\t".join([tok, "I-" + label]))
        fout.write("\n")



def _process(fout, in_file):
    """Convert column to one-per-line format."""
    import os
    with io.open(in_file, encoding="utf-8", errors="ignore") as file:
        file = json.load(file)
        
        for data in file:
            text = data["data"]["text"]
            idxs = []
            for ann in data["annotations"][0]["result"]:
                idxs.append([ann["value"]["start"], ann["value"]["end"], ann["value"]["labels"]])
            
            prev = 0
            for start, end, label in idxs:
                if start != prev:
                    toks = text[prev:start].split()
                    _write(toks, "O", fout)
                
                toks = text[start:end].split()
                _write(toks, label[0], fout)
                prev = end
            
            if end != len(text):
                toks = text[end:].split()
                _write(toks, "O", fout)

            fout.write("\n")
            

def process(in_file, out_file):
    """Convert column to one-per-line format."""
    print(f"Save to {out_file}")
    with io.open(out_file, "w", encoding="utf-8", errors="ignore") as fout:
        _process(fout, in_file)



def build_args(parser):
    """Build arguments."""
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    """Main workflow."""
    args = build_args(argparse.ArgumentParser())
    process(args.in_file, args.out_file)


if __name__ == "__main__":
    main()
