import nltk
nltk.download('stopwords')
from aspect_keybert import AspectKeyBERT
import yake
import jieba, jieba.analyse
import time
import re
from nltk.tokenize import word_tokenize


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
def get_stopwords():
    return stopwords


table = str.maketrans({"-":  r"\-", "]":  r"\]", "[":  r"\[", "\\": r"\\", \
                       "^":  r"\^", "$":  r"\$", "*":  r"\*", ".":  r"\.", \
                        "(":  r"\(", ")":  r"\)", \
                       })

class SketchExtractor:
    def __init__(self, model='yake'):
        assert model in ['yake', 'bert','jieba'], 
        self.model = model
        self.mask = '<mask>'
        self.sep = ' '
        if model == 'yake':
            self.extractor = None
        if model == 'bert':
            self.extractor = AspectKeyBERT(model='all-MiniLM-L6-v2')
        

    def get_kws(self, s, max_ngram=3, top=10, aspect_keywords=None, use_aspect_as_doc_embedding=False):
        if self.model == 'yake':
            self.extractor = yake.KeywordExtractor(n=max_ngram,top=top, windowsSize=1)
            kws_pairs = self.extractor.extract_keywords(s)
        if self.model == 'bert':
            kws_pairs = self.extractor.extract_aspect_keywords(s,top_n=top, keyphrase_ngram_range=(1,max_ngram),
                                            aspect_keywords=aspect_keywords,
                                            use_aspect_as_doc_embedding=use_aspect_as_doc_embedding,)
        return kws_pairs, [p[0] for p in kws_pairs]

    def get_sketch_from_kws(self, s, kws, template=4):
       
        mask = self.mask
        sep = self.sep
        if template == 1:
            return ' '.join(kws)
        if template == 2:
            orders = []
            remain_kws = []
            for w in kws:
                try:
                    order = s.index(w)
                    orders.append(order)
                    remain_kws.append(w)
                except:
                    pass
            kws = remain_kws
            kws_with_order = [(w,o) for w,o in zip(kws, orders)]
            kws_with_order = sorted(kws_with_order, key=lambda x:x[1])
            osrted_kws = [p[0] for p in kws_with_order]
            return ' '.join(osrted_kws)
        if template == 3:
            all_ids = []
            for w in kws: 
                try:
                    for m in list(re.finditer(re.escape(w.translate(table)),s)): 
                        all_ids += list(range(m.start(),m.end()))
                except Exception as e:
                    print(e)
                    print(w, ' |not found in| ', s)
            all_ids = sorted(list(set(all_ids)))
            masked_text = []
            for i,id in enumerate(all_ids):
                if id - all_ids[i-1] > 1: 
                    masked_text.append(' ')
                masked_text.append(s[id])
            return ''.join(masked_text)
        if template == 4:
            all_ids = []
            for w in kws:
                try:
                    for m in list(re.finditer(re.escape(w.translate(table)),s)): 
                        all_ids += list(range(m.start(),m.end()))
                except Exception as e:
                    print(e)
                    print(w, ' |not found in| ', s)
            all_ids = sorted(list(set(all_ids)))
            
            masked_text = []
            for i,id in enumerate(all_ids):
                if i == 0 and id != 0: 
                    masked_text.append(f'{mask}{sep}')
                if sep == ' ' and id - all_ids[i-1] == 2 and s[id-1] == ' ': 
                    masked_text.append(' ')
                if sep == '' and id - all_ids[i-1] == 2:
                    masked_text.append(f'{sep}{mask}{sep}')
                if id - all_ids[i-1] > 2: 
                    masked_text.append(f'{sep}{mask}{sep}')
                masked_text.append(s[id])
                if i == len(all_ids)-1 and id != len(s)-1:
                    masked_text.append(f'{sep}{mask}')
            return ''.join(masked_text)
    
    def get_sketch(self, s, max_ngram=3, top=10, aspect_keywords=None, use_aspect_as_doc_embedding=False, template=4):
        _, kws = self.get_kws(s, max_ngram, top, aspect_keywords, use_aspect_as_doc_embedding)
        sketch = self.get_sketch_from_kws(s, kws, template=template)
        return sketch
              
               

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    new_text =  re.sub(pat, '', text)
    return new_text
def remove_brakets(text):
    text =  re.sub(r'\[(.*)\]', '', text)
    text =  re.sub(r'\((.*)\)', '', text)
    return text
def clean_pipeline(text):
    return remove_brakets(remove_special_characters(text))


import torch, random
import numpy as np
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


from torch.utils.data import Dataset
class List2Dataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i]


