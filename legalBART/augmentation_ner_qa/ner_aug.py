from datasets import load_dataset, load_from_disk
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from collections import defaultdict
import random
random.seed(5)
import sys
sys.path.append('../')
from ner_utils import SketchExtractor, List2Dataset


import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='ner_train_preamble', help='dataset name in HF')
parser.add_argument('--train_size', type=int, default=400, help='labeled size')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()



raw_dataset = load_from_disk('/fs/nexus-projects/audio-visual_dereverberation/legal-data/genius/preprocessed_dataset/ner_train_pre_preamble_hf')
print('raw_dataset length',len(raw_dataset))

tag_names = ["OTHERS", "B-PETITIONER", "I-PETITIONER", "B-COURT", "I-COURT", "B-RESPONDENT", "I-RESPONDENT", "B-JUDGE", "I-JUDGE", "B-OTHER_PERSON", "I-OTHER_PERSON", "B-LAWYER", "I-LAWYER", "B-DATE", "I-DATE", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-STATUTE", "I-STATUTE", "B-PROVISION", "I-PROVISION", "B-PRECEDENT", "I-PRECEDENT", "B-CASE_NUMBER", "I-CASE_NUMBER", "B-WITNESS", "I-WITNESS"]

def get_mention_name(tag):
    return tag_names[tag].split('-')[-1]


def concat_multiple_sequences(dataset, size=3, overlap=True):
    new_dataset = defaultdict(list)
    l = len(dataset)
    if overlap:
        for i in range(l-size):
            concat_tokens = np.concatenate(dataset[i:i+size]['tokens'])
            concat_tags = np.concatenate(dataset[i:i+size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    else:  
        for i in range(l//size):
            concat_tokens = np.concatenate(dataset[i*size:(i+1)*size]['tokens'])
            concat_tags = np.concatenate(dataset[i*size:(i+1)*size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    return new_dataset


def extract_mentions(tokens, tags):
    mentions = []
    mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'OTHERS'}
    
    for i in range(len(tokens)):
        mention = get_mention_name(tags[i])
        if mention == 'OTHERS':
            continue
        if tags[i] % 2 == 1 or len(mention_dict[mention]) == 0:
            mention_dict[mention].append([tokens[i]])
            mentions.append([tokens[i]])
        else:
            mention_dict[mention][-1].append(tokens[i])
            mentions[-1].append(tokens[i])
    for k in mention_dict:
        if mention_dict[k]:
            mention_dict[k] = [' '.join(items) for items in mention_dict[k]]
    mentions = [' '.join(items) for items in mentions]
    return mentions,mention_dict


class MyTagger:
    def __init__(self, global_mention_dict):
        all_mentions = []
        for k in global_mention_dict:
            if k != 'OTHERS':
                all_mentions += global_mention_dict[k]
        all_mentions = [tuple(m.split(' ')) for m in all_mentions]
        self.all_mentions = all_mentions
        self.mwe_tokenizer = nltk.tokenize.MWETokenizer(all_mentions,separator=' ')
        self.global_mention_dict = global_mention_dict
        self.reverse_global_mention_dict = {}
        for k in global_mention_dict:
            for e in global_mention_dict[k]:
                self.reverse_global_mention_dict[e] = k

    def tokenize(self, sentence):
        return self.mwe_tokenizer.tokenize(word_tokenize(sentence))
    
    def tag(self, sentence):
        big_words = self.tokenize(sentence) 
        tags = []
        tokens = []
        for big_word in big_words:
            if big_word in self.reverse_global_mention_dict: 
                full_tag_name = self.reverse_global_mention_dict[big_word]
                for i,single_word in enumerate(word_tokenize(big_word)):
                    if i == 0: 
                        tags.append(tag_names.index('B-'+full_tag_name))
                    else: 
                        tags.append(tag_names.index('I-'+full_tag_name))
                    tokens.append(single_word)
            else:
                for single_word in word_tokenize(big_word):
                    tags.append(0)
                    tokens.append(single_word)
        assert len(tokens) == len(tags),'.'
        return tokens, tags




sketch_extractor = SketchExtractor('yake')


exp_data = raw_dataset.select(range(args.train_size))

longer_data = concat_multiple_sequences(exp_data) 
global_mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'OTHERS'}
for tokens, tags in zip(tqdm(longer_data['tokens']), longer_data['ner_tags']):
    mentions, mention_dict = extract_mentions(tokens, tags)
    for k in mention_dict:
        global_mention_dict[k] += mention_dict[k]
        global_mention_dict[k] += [s.lower() for s in mention_dict[k]]
        global_mention_dict[k] += [s.upper() for s in mention_dict[k]]
        global_mention_dict[k] += [s.title() for s in mention_dict[k]]
        global_mention_dict[k] += [s.capitalize() for s in mention_dict[k]]

for k in global_mention_dict: 
    global_mention_dict[k] = list(set(global_mention_dict[k]))
my_tagger = MyTagger(global_mention_dict)



sketches = []
for tokens, tags in zip(tqdm(longer_data['tokens']), longer_data['ner_tags']):
    text = ' '.join(tokens)
    
    _, kws = sketch_extractor.get_kws(text, max_ngram=3, top=max(len(tokens)//8,5))
    mentions, _ = extract_mentions(tokens, tags)
    sketch = sketch_extractor.get_sketch_from_kws(text, kws+mentions)
    
    sketches.append(sketch)

sketch_dataset = List2Dataset(sketches)

genius = pipeline('text2text-generation',model='/fs/nexus-projects/audio-visual_dereverberation/legal-data/genius/saved_models/bart-large-ner_train-sketch4/checkpoint-1720',device=0)



augmented_dataset = defaultdict(list)


for i in range(args.n_aug):
    for out in tqdm(genius(sketch_dataset, num_beams=3, do_sample=True, max_length=100, batch_size=32)):
        generated_text = out[0]['generated_text']
        new_tokens, new_tags = my_tagger.tag(generated_text)
        augmented_dataset['tokens'].append(new_tokens)
        augmented_dataset['ner_tags'].append(new_tags)
        
print(f'>>> Num of generated examples: {len(augmented_dataset["tokens"])}')

print(f'filtering too short seqs...')
tokens_list = []
tags_list = []
for tokens,tags in zip(augmented_dataset['tokens'], augmented_dataset['ner_tags']):
    if len(list(set(tags))) > 1 and len(tags) > 5: 
        tokens_list.append(tokens)
        tags_list.append(tags)

augmented_dataset = {'tokens':tokens_list, 'ner_tags':tags_list}
print(f'>>> Num of filtered examples: {len(augmented_dataset["tokens"])}')


df = pd.DataFrame(augmented_dataset)
file_name = f'../ner_data_pkl/{args.dataset_name}-{args.train_size}-naug-{args.n_aug}.pkl'
df.to_pickle(file_name)  
print(file_name)
