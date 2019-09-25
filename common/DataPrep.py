"""
Data preparation file
"""

import numpy as np
import pandas as pd

def clean_string(string):
    return ''.join(e for e in string if e.isalnum())

def reverse_dict(d):
    new_d = {}
    for k,i in d.items():
        new_d[i] = k
    return new_d

def mapping(files):
    """
    Creates a unique index for ever entity and relation.
    """
    E = set()
    R = set()
    
    for filename in files:           
        with open(filename,'r') as f:
            for line in f:
                s,p,o = line.split()
                E.add(s)
                E.add(o)
                R.add(p)
    
    E_mapping = {e:i for i,e in enumerate(E)}
    R_mapping = {e:i for i,e in enumerate(R)}
    
    return E_mapping, R_mapping

def prep_data(filename):
    """
    Prep data without labels
    """
    
    sub = []
    obj = []
    pred = []
    with open(filename, 'r') as f:
        for l in f:
            s,p,o = l.split()
            sub.append(s.strip())
            obj.append(o.strip())
            pred.append(p.strip())
    
    return sub, obj, pred


def class_distribution(filename, R_mapping):
    """
    Return the a priori probability for each class.
    """
    _, _, pred = prep_data(filename)
    labels = np.asarray([R_mapping[i] for i in pred])
    unique, counts = np.unique(labels, return_counts=True)
    keys = list(R_mapping.keys())
    return_dict = {keys[i]:k/len(labels) for i,k in zip(unique, counts)}
    return return_dict
    
def oneToOne(Xss, Xoo, Xpp, E_mapping, R_mapping):
    return np.asarray(Xss), np.asarray(Xoo), np.asarray(Xpp)
 
def NToN(Xss, Xoo, Xpp, E_mapping, R_mapping):
    Xs = []
    Xo = []
    Xp = []
    
    for k in list(R_mapping.keys()):
        idx = np.where(Xpp == R_mapping[k])
        Xs.append(Xss[idx])
        Xo.append(Xoo[idx])
        Xp.append([R_mapping[k]])
        
    return np.asarray(Xs), np.asarray(Xo), np.asarray(Xp)
    
def oneToN(Xss, Xoo, Xpp, E_mapping, R_mapping):
    Xs = []
    Xp = []
    Xo = []
    
    for e in list(E_mapping.keys()):
        idx_e = np.where(Xss == E_mapping[e])
        for r in list(R_mapping.keys()):
            idx_r = np.where(Xpp == R_mapping[r])
            idx = np.intersect1d(idx_e, idx_r)
            if len(idx) > 0:
                Xs.append([E_mapping[e]])
                Xp.append([R_mapping[r]])
                Xo.append(Xoo[idx])

    return np.asarray(Xs), np.asarray(Xo), np.asarray(Xp)

def NToOne(Xss, Xoo, Xpp, E_mapping, R_mapping):
    Xs = []
    Xp = []
    Xo = []
    
    for e in list(E_mapping.keys()):
        idx_e = np.where(Xoo == E_mapping[e])
        for r in list(R_mapping.keys()):
            idx_r = np.where(Xpp == R_mapping[r])
            idx = np.intersect1d(idx_e, idx_r)
            if len(idx) > 0:
                Xo.append([E_mapping[e]])
                Xp.append([R_mapping[r]])
                Xs.append(Xoo[idx])
    
    return np.asarray(Xs), np.asarray(Xo), np.asarray(Xp)
 
def apply_mapping(dataframe, E_mapping, R_mapping, mode = '1-to-1'):
    Xs = dataframe.iloc[:,0]
    Xp = dataframe.iloc[:,1]
    Xo = dataframe.iloc[:,2]
    
    Xs = np.asarray([E_mapping[str(i)] for i in Xs], dtype = np.float32).reshape((-1,1))
    Xo = np.asarray([E_mapping[str(i)] for i in Xo], dtype = np.float32).reshape((-1,1))
    Xp = np.asarray([R_mapping[str(i)] for i in Xp], dtype = np.float32).reshape((-1,1))
    
    if mode == '1-to-1':
        Xs,Xo,Xp = oneToOne(Xs, Xo, Xp, E_mapping, R_mapping)
    elif mode == '1-to-N': 
        Xs,Xo,Xp = oneToN(Xs, Xo, Xp, E_mapping, R_mapping)
    elif mode == 'N-to-N': 
        Xs,Xo,Xp = NToN(Xs, Xo, Xp, E_mapping, R_mapping)
    elif mode == 'N-to-1': 
        Xs,Xo,Xp = NToOne(Xs, Xo, Xp, E_mapping, R_mapping)
    else:
        raise NotImplementedError(mode)
    
    return Xs, Xp, Xo
 
def data_iterator(filename, E_mapping, R_mapping, batch_size = 128, mode = '1-to-1'):
    """
    Creates batches for the large data set.
    
    args:
        filename: file containing triples.
        E_mapping: relation between entity and index in vocabulary.
        R_mapping: relation between relation and index in vocabulary.
        mode: different scoring methods. 1-to-1 scores every triple by itself. 1-to-N scores all objects for every subject, predicate pair. N-to-N scores all subjects and objects for every relation, this is the fastest method, however, may give inaccurate training. N-to-1 is the reverse of 1-to-N.
        passes: number of passes through the data.
        
    returns:
        Xs: (batch_size, 1) array, index for subjects
        Xo: (batch_size, 1) array
        Xp: (batch_size, 1) array
        probs: (batch_size, 1) array, labels for triples.
    """
    
    #VOC_SIZE = len(list(E_mapping.keys()))
    #REL_SIZE = len(list(R_mapping.keys()))
    
    if batch_size == -1:
        batch_size = 10**6
    
    tmp = pd.read_csv(filename, sep='\t', header = None, low_memory = False, dtype=str)
    
    if tmp.shape[-1] < 3:
        sep = ' '
    else:
        sep = '\t'
    
    pa = 0
    try:
        for data in pd.read_csv(filename, sep=sep, header = None, low_memory = False, chunksize = batch_size, dtype=str): 
            yield apply_mapping(data, E_mapping, R_mapping, mode = mode)
    except ValueError:
        return []
    
def save_obj(obj, name):
    with open('tmp/'+ name + '.txt', 'w') as f:
        for k,i in obj.items():
            print(k, i, file = f)

def load_obj(name):
    d = {}
    with open('tmp/' + name + '.txt', 'r') as f:
        for l in f:
            k,i = l.split()
            d[k] = int(i)
    return d
    
def save_embedding(weights, name):
    d = {}
    k = 0
    for w in weights:
        d[k] = w
        k += 1
    save_obj(d, name)

def load_embedding(name):
    weights = []
    d = load_obj(name)
    od = collections.OrderedDict(sorted(d.items()))
    for _, v in od.items(): 
        weights.append(v)
    return np.asarray(weights).reshape((-1, len(list(d.keys()))))
        
