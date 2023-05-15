fpath = 'eng.train.iobes.txt'
ofpath = 'eng.linearized.train.iobes.txt'
def preprocess(sentence):
    original = ''
    tmp = []
    string = []
    dict1 = {}
    n = 0
    s=""
    v=""
    for w in sentence:
        original = ' '.join([w[0] for w in sentence])
    return original
    
with open(fpath, 'r') as inf, open(ofpath, 'w') as of:
    counter = 0
    sentence = []
    for line in inf:     
        line = line.strip()
        if line != '':
            line = line.split()
            if len(line) == 2:
                sentence.append(line)
        else:
            original = preprocess(sentence)
            of.write(original + '\n')
            sentence = []
            counter+=1