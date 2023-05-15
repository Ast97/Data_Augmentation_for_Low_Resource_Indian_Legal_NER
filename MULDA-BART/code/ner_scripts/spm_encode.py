import sentencepiece as spm


sp = spm.SentencePieceProcessor(model_file='./MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/sentence.bpe.model')

with open('./MULDA/data/indian_ner_legal_dataset/linearized_data/train_linearized.txt', 'r', encoding="utf-8") as rf, open('./MULDA/data/indian_ner_legal_dataset/spm_encoded/train_spm.eng', 'w', encoding="utf-8") as wf:
    for line in rf:
        wf.write(' '.join(sp.encode(line, out_type=str)))
        wf.write("\n")


with open('./MULDA/data/indian_ner_legal_dataset/linearized_data/test_linearized.txt', 'r', encoding="utf-8") as rf, open('./MULDA/data/indian_ner_legal_dataset/spm_encoded/test_spm.eng', 'w', encoding="utf-8") as wf:
    for line in rf:
        wf.write(' '.join(sp.encode(line, out_type=str)))
        wf.write("\n")

with open('./MULDA/data/indian_ner_legal_dataset/linearized_data/val_linearized.txt', 'r', encoding="utf-8") as rf, open('./MULDA/data/indian_ner_legal_dataset/spm_encoded/val_spm.eng', 'w', encoding="utf-8") as wf:
    for line in rf:
        wf.write(' '.join(sp.encode(line, out_type=str)))
        wf.write("\n")


