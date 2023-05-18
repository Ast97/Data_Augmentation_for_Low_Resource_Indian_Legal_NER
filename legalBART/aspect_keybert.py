from keybert import KeyBERT
from keybert.backend._utils import select_backend
import numpy as np
import re
from typing import List, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
basic_stops = stopwords.words('english')
basic_stops = [w for w in basic_stops if "n't" not in w]
basic_stops = [w for w in basic_stops if w not in ['no','not','nor','but']]
stopwords = basic_stops[:]
stopwords += [w.capitalize() for w in basic_stops]
stopwords += [w.upper() for w in basic_stops]


class AspectKeyBERT(KeyBERT):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = select_backend(model)
    def extract_aspect_keywords(
        self,
        doc: str,
        use_aspect_as_doc_embedding: bool = False,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        aspect_keywords: List[str] = None,
    ) -> List[Tuple[str, float]]:
        
        try:
            if candidates is None:
                if vectorizer:
                    count = vectorizer.fit([doc])
                else:
                    count = CountVectorizer(
                        ngram_range=keyphrase_ngram_range, lowercase=False, tokenizer=word_tokenize, token_pattern=None
                    ).fit([doc])
                candidates = count.get_feature_names()
                def there_is_punc(text):
                    return len(re.findall(r'[，。：；‘’“”;,\.\?\!\(\)\[\]\{\}:\'\"\-\@\#\$\%\^\&\*]',text))
                candidates = [c for c in candidates if not there_is_punc(c)]

            if use_aspect_as_doc_embedding:
                assert aspect_keywords is not None, 
                doc_embedding = self.model.embed([" ".join(aspect_keywords)])
            else:
                doc_embedding = self.model.embed([doc])
            candidate_embeddings = self.model.embed(candidates)

            if aspect_keywords is not None and use_aspect_as_doc_embedding == False:
                aspect_embeddings = self.model.embed([" ".join(aspect_keywords)])
                doc_embedding = np.average(
                    [doc_embedding, aspect_embeddings], axis=0, weights=[1, 1]
                )


            if use_mmr:
                keywords = mmr(
                    doc_embedding, candidate_embeddings, candidates, top_n, diversity
                )
            elif use_maxsum:
                keywords = max_sum_similarity(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    nr_candidates,
                )
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [
                    (candidates[index], round(float(distances[0][index]), 4))
                    for index in distances.argsort()[0][-top_n:]
                ][::-1]

            return keywords
        except ValueError:
            return []
