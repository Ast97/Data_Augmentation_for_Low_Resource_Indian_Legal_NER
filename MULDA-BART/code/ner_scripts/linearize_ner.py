import pandas as pd
import json, io

train_df_judgement = pd.read_json('./MULDA/data/indian_ner_legal_dataset/raw_train/NER_TRAIN_JUDGEMENT.json')
train_df_preamble = pd.read_json('./MULDA/data/indian_ner_legal_dataset/raw_train/NER_TRAIN_PREAMBLE.json')

test_df_judgement = pd.read_json('./MULDA/data/indian_ner_legal_dataset/raw_test/NER_DEV_JUDGEMENT.json')
test_df_preamble = pd.read_json('./MULDA/data/indian_ner_legal_dataset/raw_test/NER_DEV_PREAMBLE.json')

total_data = len(train_df_judgement) + len(train_df_preamble) + len(test_df_judgement) + len(test_df_preamble)

df = pd.concat([train_df_judgement, train_df_preamble])
df.reset_index(inplace=True, drop=True)

split=int(total_data*0.9)

train_df = df[:split]
val_df = df[split:]

test_df = pd.concat([test_df_judgement, test_df_preamble])

#train_df = train_df.head(10)
#test_df = test_df.head(10)
#val_df = val_df.head(10)

max_sentence = 0

with io.open("./MULDA/data/indian_ner_legal_dataset/linearized_data/train_linearized.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in train_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")

with io.open("./MULDA/data/indian_ner_legal_dataset/linearized_data/test_linearized.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in test_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")

with io.open("./MULDA/data/indian_ner_legal_dataset/linearized_data/val_linearized.txt", "w", encoding="utf-8", errors="ignore") as fout:
    for txt in val_df['data'].values:
        val = dict(txt)['text']
        if len(val) > max_sentence:
            max_sentence = len(val)
        fout.write(f"{val}\n")


print('Pre-Processing Data for Train, Test and Val. Max Tokens - ', len(train_df), len(test_df),len(val_df), max_sentence)