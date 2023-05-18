import argparse
import json
import numpy as np
import torch
import pandas as pd
import transformers
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

def calculatePerplexity(sentence,model,tokenizer):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) 
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)

model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to('cuda')
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

df = pd.read_csv('/legal-data/augmented_data/legal_BART_Samples.csv')
tokens = df['tokens']

sum = 0
for i in range(len(tokens)):
    perplexity = calculatePerplexity(tokens[i],model,tokenizer)
    sum += perplexity
print(sum/len(tokens))