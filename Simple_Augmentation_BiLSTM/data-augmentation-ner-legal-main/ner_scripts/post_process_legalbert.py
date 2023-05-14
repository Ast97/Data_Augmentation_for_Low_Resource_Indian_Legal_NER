import os, csv

label_list_encoding_dict = {0: "OTHERS", 1: "PETITIONER", 2: "COURT", 3: "RESPONDENT", 4: "JUDGE", 5: "OTHER_PERSON", 6: "LAWYER", 7: "DATE", 8: "ORG", 9: "GPE", 10: "STATUTE", 11: "PROVISION", 12: "PRECEDENT", 13: "CASE_NUMBER",14:"WITNESS"}
ner_tag_encoding = {"OTHERS" : 0, "PETITIONER" : 1, "COURT" : 2, "RESPONDENT" : 3, "JUDGE" : 4, "OTHER_PERSON" : 5, "LAWYER" : 6, "DATE" : 7, "ORG" : 8, "GPE" : 9, "STATUTE" : 10, "PROVISION" : 11, "PRECEDENT" : 12, "CASE_NUMBER" : 13, "WITNESS" : 14}

def post_process_dataaugmentations(fin, fout):
    with open(fin, 'r', encoding="utf-8") as f:
        dataset = f.readlines()
    
    lines = []
    tokens = []
    ner_tags = []
    for line in dataset:
        if line == '\n':
            lines.append([tokens, ner_tags])
            tokens = []
            ner_tags = []
        else:
            token, tag = line.split(' ')
            token, tag = token.strip(), tag.strip()
            tokens.append(token)
            if tag == 'O':
                ner_tags.append(ner_tag_encoding['OTHERS'])
            elif tag[0] == 'B' or tag[0] == 'I':
                tag = tag[2:]
                if tag in ner_tag_encoding:
                    ner_tags.append(ner_tag_encoding[tag])
                else:
                    print('Something is wrong!')
                    exit()
            else:
                print('Something is Wrong2')
                exit()
    with open(fout, 'w', encoding="utf-8") as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        f.write('tokens,ner_tags\n')
        for line in lines:
            csv_writer.writerow(line)
            #f.write("\""+str(line[0])+'\"' +"," +"\""+str(line[1])+'\"\n')
        f.close()


train_in = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/train.txt'
test_in = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/test.txt'
dev_in = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/dev.txt'

train_out = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/train_postprocessed.csv'
test_out = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/test_postprocessed.csv'
dev_out = './data-augmentation-ner-legal-main/src/datasets/cMR1.0/dev_postprocessed.csv'

post_process_dataaugmentations(train_in, train_out)
post_process_dataaugmentations(test_in, test_out)
post_process_dataaugmentations(dev_in, dev_out)

#Sanity Check
import pandas as pd
SimpleAug_df = pd.read_csv('./data-augmentation-ner-legal-main/src/datasets/cMR1.0/train_postprocessed.csv')
print(SimpleAug_df['tokens'][0])
print(SimpleAug_df['ner_tags'][0])
print(len(eval(SimpleAug_df['tokens'][0])))
print(len(eval(SimpleAug_df['ner_tags'][0])))