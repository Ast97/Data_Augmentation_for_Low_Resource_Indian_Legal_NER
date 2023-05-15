import pandas as pd
import json, io, os


train_df_judgement = pd.read_csv('./data/bio_format/NER_TRAIN_JUDGEMENT.txt')
train_df_preamble = pd.read_csv('./data/bio_format/NER_TRAIN_PREAMBLE.txt')

test_df_judgement = pd.read_csv('./data/bio_format/NER_DEV_JUDGEMENT.txt')
test_df_preamble = pd.read_csv('./data/bio_format/NER_DEV_PREAMBLE.txt')

total_data = len(train_df_judgement) + len(train_df_preamble) + len(test_df_judgement) + len(test_df_preamble)

df = pd.concat([train_df_judgement, train_df_preamble])
df.reset_index(inplace=True, drop=True)

split=int(total_data*0.75)

train_df = df[:split]
val_df = df[split:]

test_df = pd.concat([test_df_judgement, test_df_preamble])

#train_df = train_df.head(10)
#test_df = test_df.head(10)
#val_df = val_df.head(10)

max_sentence = 0

os.mkdir('./src/datasets/___1.1')

with io.open("./src/datasets/___1.1/train.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in train_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")

with io.open("./src/datasets/___1.1/test.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in test_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")

with io.open("./src/datasets/___1.1/dev.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in val_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")


print('Pre-Processing Data for Train, Test and Val. Max Tokens - ', len(train_df), len(test_df),len(val_df), max_sentence)