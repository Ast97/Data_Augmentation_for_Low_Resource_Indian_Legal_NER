import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from datasets import load_dataset, load_from_disk
nltk.download('stopwords')
nltk.download('punkt')
import random
random.seed(5)
import sys
import os
sys.path.append('/fs/nexus-projects/audio-visual_dereverberation/legal-data/genius/')

print(sys.path)
from ner_utils import SketchExtractor, clean_pipeline


import os
os.system('export HF_DATASETS_CACHE="../saved_datasets/hf_cache"')
os.system('export HF_DATASETS_OFFLINE=1')

names = [
            "ner_dev"          
        ]
for name in names:
    raw_dataset =  load_dataset('json', data_files= 'NER_DEV.zip')
    print("Working on dataset : " + name)
    print(raw_dataset)
    print(raw_dataset["train"]["data"][0])


    sketch_extractor = SketchExtractor(model='yake')

    def a_len(s):
        return len(s.split(' ')) 


    def text_preprocess(examples):
        res = defaultdict(list)
        documents = []
        for data in examples['data']:
            documents.append(data['text'])

        for document in documents:
            document = clean_pipeline(document)
            document = document.replace('\n',' ').replace('  ',' ')
            document = document.replace('\'s','')
            sents = sent_tokenize(document)
            res['passage'].append(' '.join(sents[:15]))
            
        return res

    preporcessed_dataset = raw_dataset.map(text_preprocess,batched=True,\
                                remove_columns=raw_dataset['train'].column_names,\
                                batch_size=128, num_proc=128)

    print("$ck pre-processed dataset for : " + name) 
    print(preporcessed_dataset)

    def add_sketch_to_dataset(examples):
        res = defaultdict(list)
        passages = examples['passage']

        for p in passages:
            
            res['text'].append(p)
            _, kws = sketch_extractor.get_kws(p, max_ngram=3, top=max(a_len(p)//5,1)) # max 3-gram
            sketch = sketch_extractor.get_sketch_from_kws(p, kws, template=4)
            res['sketch_4'].append(sketch)
        return res

    dataset_with_sketch = preporcessed_dataset.map(add_sketch_to_dataset, batched=True, 
                                                    remove_columns=preporcessed_dataset['train'].column_names,
                                                    batch_size=8, num_proc=256)
    print("$ck dataset_with_sketch for : " + name)

    print(dataset_with_sketch)

    dataset_with_sketch.save_to_disk(f'/fs/nexus-projects/audio-visual_dereverberation/legal-data/genius/saved_datasets/{name}/')