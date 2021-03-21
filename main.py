import pickle
# import pandas as pd
import numpy as np
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
def callers(sen):
    
    label_cols = ['toxic', 'obscene', 'insult']
    preds = np.zeros((1, len(label_cols)))
    with open('M.pickle', 'rb') as fm:
        M = pickle.load(fm)

    with open('vec.pickle', 'rb') as fm:
        vec = pickle.load(fm)

    print('Processing.......')
    sen=[sen]
    test_term_doc = vec.transform(sen)
    test_x = test_term_doc
    for i , j in enumerate(label_cols):
        m,r = M[i]
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    print(preds[0][0]*100)
    print(preds[0][1]*100)
    print(preds[0][2]*100)
    return preds[0][0]*100,preds[0][1]*100,preds[0][2]*100
callers('welcome!!')