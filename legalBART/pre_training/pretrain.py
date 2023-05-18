import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--sketch_type', type=str, default=4, help='1,2,3 or 4. Details see paper')
args = parser.parse_args()

# pretrained checkpoint:
model_checkpoint = '/saved_models/bart-large-ner_train-sketch4/checkpoint-190105'  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("$ck max length : " + str(tokenizer.model_max_length))
##################################################################
#                     data pre-processing
##################################################################


from datasets import load_from_disk
dataset_path = '/saved_datasets/ner_train' 
dataset_name = dataset_path.split('/')[-1]
dataset_with_sketch = load_from_disk(dataset_path)

print(dataset_with_sketch)

max_input_length = 1024
max_target_length = 1024
print("********** Sketch type is: ", args.sketch_type)
def preprocess_function(examples):
    """
    # inputs: the sketch
    # labels: the original text
    """
    model_inputs = tokenizer(examples[f'sketch_{args.sketch_type}'], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset_with_sketch.map(preprocess_function, batched=True, 
                                         batch_size=10000,num_proc=100)


rouge_score = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

##################################################################
#                     training
##################################################################

batch_size = 8
num_train_epochs = 5
model_name = model_checkpoint.split("/")[-1]


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


logging_steps = len(tokenized_dataset['train']) // batch_size

output_dir = f"/saved_models/bart-large-{dataset_name}-sketch{args.sketch_type}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps = 5000,      
    save_strategy = 'epoch',
    save_total_limit = 2,
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


tokenized_dataset["train"] = tokenized_dataset["train"].remove_columns(dataset_with_sketch["train"].column_names)
tokenized_dataset["validation"] = tokenized_dataset["validation"].remove_columns(dataset_with_sketch["validation"].column_names)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"], 
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
