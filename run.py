from __future__ import division, print_function, unicode_literals
import os
import io
import re
import json
import numpy as np
import scipy as sp
from lda import LDA

DEBUG = False
REGEX = r"([^\W0-9]|['_])+"
COMPILED_REGEX = re.compile(REGEX, re.UNICODE)

def tokenize(text, stopwords=set()):
    REGEX = r"([^\W0-9]|['_])+"
    COMPILED_REGEX = re.compile(REGEX, re.UNICODE)
    features = []
    for match in COMPILED_REGEX.finditer(text):
        token = match.group().lower()
        if token not in stopwords and len(token) > 3:
            features.append(token)
    return features

def state_of_the_union():
    STOPWORDS_FILE = 'english_all.txt'
    stopwords = set()
    with io.open(STOPWORDS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for tok in tokenize(f.read()):
            stopwords.add(tok)
    
    singletons = set()
    singleton_test = {}
    PATH = 'state_of_the_union/speech_files'
    seqs = []
    for root, dirs, files in os.walk(PATH):
        #~ print(files)
        for file_name in files:
            with io.open(os.path.join(root, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                seq = []
                for tok in tokenize(f.read(), stopwords):
                    singleton_test[tok] = singleton_test.setdefault(tok, 0) + 1
                    seq.append(tok)
                seqs.append(seq)
    for s, count in singleton_test.iteritems():
        if count <= 2:
            singletons.add(s)
    singleton_test = None
    
    for seq in seqs:
        new_seq = []
        for tok in seq:
            if tok not in singletons:
                new_seq.append(tok)
        yield new_seq

def test_data():
    l = [
        [1, 2, 3, 4, 5, ],
        [1, 2, 3, 6, 7, ],
        [1, 2, 3, 6, 7, ],
        [4, 5, 8, 9, ],
    ]
    #~ l = [
        #~ [1,2,3,4,5,],
        #~ [1,2,3,4,5,],
        #~ [6,7,8,9,10,],
        #~ [6,7,8,9,10,],
        #~ [1,2,3,4,5,],
    #~ ]
    for d in l:
        yield d


if __name__ == "__main__":
    # Format: (name of analysis, number of topics, alpha, beta, burn, length, dataset feature vector iterator)
    given = [
        #~ ("test", 2, 0.1, 0.1, 100, 10, test_data),
        ('state_of_the_union', 50, 0.1, 0.1, 199, 1, state_of_the_union),
    ]
    
    for settings in given:
        analysis = LDA(settings[1], settings[2], settings[3], settings[4], settings[5])
        print(settings[0])
        analysis.run_analysis(settings[6]())
        analysis.print_topics(10)
        with io.open('results_%s.json'%(settings[0]), 'w', encoding='utf-8', errors='ignore') as f:
            f.write(unicode(json.dumps(analysis.log_likelihoods)))
        
        
        
