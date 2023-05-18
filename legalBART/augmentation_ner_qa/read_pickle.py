import pickle
import pandas as pd
objects = []


with (open("/ner_data_pkl/ner_train_preamble-400-naug-1.pkl", "rb")) as openfile:
    data = pickle.load(openfile)


df = pd.DataFrame(data, columns=['tokens', 'ner_tags'])

df.to_csv('/augmented_data/ner_preamble.csv', index=False)