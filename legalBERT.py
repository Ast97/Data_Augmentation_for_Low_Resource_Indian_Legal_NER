import torch
import os
import itertools
import pandas as pd
import numpy as np
import json
import argparse
from nervaluate import Evaluator # !pip install nervaluate

import warnings
warnings.filterwarnings('ignore')


from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification


LABELS_LIST = ["OTHERS", 
            "PETITIONER", 
            "COURT", 
            "RESPONDENT", 
            "JUDGE", 
            "OTHER_PERSON", 
            "LAWYER", 
            "DATE", 
            "ORG", 
            "GPE", 
            "STATUTE", 
            "PROVISION", 
            "PRECEDENT", 
            "CASE_NUMBER", 
            "WITNESS"]

LABEL_ENCODING_DICT = {i:i for i in range(15)}


def create_datasets(args):
    train_df_judgement = pd.read_json(args.train_judgement_file)
    train_df_preamble = pd.read_json(args.train_preamble_file)

    test_df_judgement = pd.read_json(args.dev_judgement_file)
    test_df_preamble = pd.read_json(args.dev_preamble_file)

    augmented_samples = pd.read_csv(args.augmentation_samples)
    augmented_samples['tokens'] = augmented_samples['tokens'].map(lambda x: eval(x))
    augmented_samples['ner_tags'] = augmented_samples['ner_tags'].map(lambda x: eval(x))

    df = pd.concat([train_df_judgement, train_df_preamble])
    df.reset_index(inplace=True, drop=True)

    total_data = len(df) + len(test_df_judgement) + len(test_df_preamble)

    split = int(total_data * args.train_split)
    split_train  = split /(len(train_df_judgement) + len(train_df_preamble))
    train_df = df.sample(frac = split_train)

    val_df = df.drop(train_df.index)
    
    augment_size = int(len(train_df) * args.augmentation_ratio)
    train_df = pd.concat([train_df, augmented_samples[:augment_size]])
    train_df.drop('id', axis=1, inplace=True)
    # test data (appx 10% of the total data)
    test_df = pd.concat([test_df_judgement, test_df_preamble])


    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset

def tokenize_all_labels(rows, tokenizer):
    tokenized_inputs = tokenizer(list(rows["tokens"]), truncation = True, is_split_into_words = True)
    labels, label_all = [], True
    for index, label in enumerate(rows["ner_tags"]):
        prior_idx = None
        word_ids = tokenized_inputs.word_ids(batch_index = index)
        
        label_ids = []
        for current_idx in word_ids:
            if current_idx is None: label_ids.append(-100)
            elif label[current_idx] == '0': label_ids.append(0)
            elif current_idx != prior_idx: label_ids.append(LABEL_ENCODING_DICT[label[current_idx]])
            else: label_ids.append(LABEL_ENCODING_DICT[label[current_idx]] if label_all else -100)
            prior_idx = current_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def transform_data(preds):
    res = []

    for pred in preds:
        transform_pred = []
        prev_start, prev_tag = 0, pred[0]
        for id, tag in enumerate(pred[1:], 1):
            if prev_tag != "OTHERS":
                if tag == "OTHERS":
                    transform_pred.append({"label":prev_tag, "start":prev_start, "end":id-1})
                    prev_start = id
                    prev_tag = tag
                elif prev_tag != tag:
                    transform_pred.append({"label":prev_tag, "start":prev_start, "end":id-1})
                    prev_start = id
                    prev_tag = tag
            else:
                prev_start = id
                prev_tag = tag
        res.append(transform_pred)
    
    return res



def ner_evals(pred, true):

    evaluator = Evaluator(true, pred, tags=LABELS_LIST)

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    return results | results_per_tag

def evaluate_metrics(pred_tuple, metric):
    predictions, labels = pred_tuple
    predictions = np.argmax(predictions, axis=2)

    actual_predictions = [[LABELS_LIST[pred_tuple] for (pred_tuple, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    actual_labels = [[LABELS_LIST[l] for (pred_tuple, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=actual_predictions, references=actual_labels)
    results = metric.compute(predictions=actual_predictions, references=actual_labels)
    ner_results = ner_evals(transform_data(actual_predictions), transform_data(actual_labels))
    return ner_results | {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    


def main():
    
    args = build_args(argparse.ArgumentParser())

    # Get the GPU device name.
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        device_name = torch.cuda.get_device_name()
        print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    else:
        print("No gpu found; running on cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset, val_dataset, test_dataset = create_datasets(args)

    print("Created Datasets")

    # tokenizing and loading legalBERT model 
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=len(LABELS_LIST))
    metric = load_metric("seqeval")

    train_dataset_tokenized = train_dataset.map((lambda x: tokenize_all_labels(x, tokenizer=tokenizer)), batched=True)
    val_dataset_tokenized = val_dataset.map((lambda x: tokenize_all_labels(x, tokenizer=tokenizer)), batched=True)

    print("tokenised datasets")

    training_arguments = TrainingArguments(
        args.model_path + args.model_name,
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        report_to="none"
    )
    # training on train set and evaluating on validation set
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        training_arguments,
        train_dataset = train_dataset_tokenized,
        eval_dataset = val_dataset_tokenized,
        data_collator = data_collator,
        tokenizer=tokenizer,
        compute_metrics=(lambda x: evaluate_metrics(x, metric=metric))
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.model_path + args.model_name + ".model")

    print("completed training")

    test_dataset_tokenized = test_dataset.map((lambda x: tokenize_all_labels(x, tokenizer=tokenizer)), batched=True)
    test_results = trainer.evaluate(test_dataset_tokenized)
    print(test_results)


def build_args(parser):
    """Build arguments."""
    parser.add_argument("--augmentation_samples", type=str, required=True)
    parser.add_argument("--train_preamble_file", type=str, default='./Data_preprocess/NER_TRAIN/NER_TRAIN_PREAMBLE_PREPROCESSED.json')
    parser.add_argument("--train_judgement_file", type=str, default='./Data_preprocess/NER_TRAIN/NER_TRAIN_JUDGEMENT_PREPROCESSED.json')
    parser.add_argument("--dev_preamble_file", type=str, default='./Data_preprocess/NER_DEV/NER_DEV_PREAMBLE_PREPROCESSED.json')
    parser.add_argument("--dev_judgement_file", type=str, default='./Data_preprocess/NER_DEV/NER_DEV_JUDGEMENT_PREPROCESSED.json')
    parser.add_argument("--augmentation_ratio", type=float, default=0.25)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.910635913133073e-05)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--checkpoint_path", type=str, default="./models/checkpoints/")
    parser.add_argument("--model_path", type=str, default="./models/")
    parser.add_argument("--model_name", type=str, default="indian_legal_ner")
    return parser.parse_args()

if __name__ == "__main__":
    main()