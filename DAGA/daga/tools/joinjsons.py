import json
import pandas as pd
import argparse

def joinjsons(file1, file2, out):
    with open(file1) as f1:               # open the file
        data1 = json.load(f1)

    with open(file2) as f2:                # open the file       
        data2 = json.load(f2)
        
    # df1 = pd.DataFrame([data1])                      # Creating DataFrames
    # df2 = pd.DataFrame([data2])                      # Creating DataFrames
    # print(df1[0], df2)
    # MergeJson = pd.concat([df1, df2], axis=0)         # Concat DataFrames

    # print(data1+data2)
    with open(out, "w") as write_file:
        json.dump(data1+data2, write_file)


def build_args(parser):
    """Build arguments."""
    parser.add_argument("--in_file1", type=str, default="../../NER_TRAIN/NER_TRAIN_PREAMBLE.json")
    parser.add_argument("--in_file2", type=str, default="../../NER_TRAIN/NER_TRAIN_JUDGEMENT.json")
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = build_args(argparse.ArgumentParser())
    joinjsons(args.in_file1, args.in_file2, args.out_file)

if __name__ == "__main__":
    main()
