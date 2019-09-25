"""
main.py
"""
import tensorflow as tf
import numpy as np
import scipy as sp
import argparse
import os

from random import choice
from tqdm import tqdm

from common.DataPrep import mapping, save_obj, load_obj, data_iterator, clean_string
from common.FalseTriples import corrupt_triples

from keras.constraints import MaxNorm, UnitNorm, NonNeg
from keras.regularizers import l1,l2,l1_l2

from collections import defaultdict
from keras import backend as K

def reverse_dict(d):
    return {i:k for k,i in d.items()}

def store_embedding(model, mapping_e, mapping_r, directory):
    
    mapping_e = reverse_dict(mapping_e)
    mapping_r = reverse_dict(mapping_r)
    
    if hasattr(model, 'e'):
        a = np.squeeze(np.asarray(model.e.get_weights()))
        with open(directory + '/' + 'entity_embedding' + '.txt','w') as f:
            for i,v in enumerate(a):
                f.write(str(mapping_e[i])+'|'+','.join([str(w) for w in v]) + '\n')
        
    if hasattr(model, 'e_complex'):
        a = np.squeeze(np.asarray(model.e_complex.get_weights()))
        with open(directory + '/' + 'complex_entity_embedding' + '.txt','w') as f:
            for i,v in enumerate(a):
                f.write(str(mapping_e[i])+'|'+','.join([str(w) for w in v]) + '\n')
                
    if hasattr(model, 'r'):
        a = np.squeeze(np.asarray(model.r.get_weights()))
        with open(directory + '/' + 'relation_embedding' + '.txt','w') as f:
            for i,v in enumerate(a):
                f.write(str(mapping_r[i])+'|'+','.join([str(w) for w in v]) + '\n')
                
    if hasattr(model, 'r_complex'):
        a = np.squeeze(np.asarray(model.r_complex.get_weights()))
        with open(directory + '/' + 'complex_relation_embedding' + '.txt','w') as f:
            for i,v in enumerate(a):
                f.write(str(mapping_r[i])+'|'+','.join([str(w) for w in v]) + '\n')
                

def get_all_triples(filename, E_mapping, R_mapping):
    tmp_s = []
    tmp_o = []
    tmp_p = []
    for true_vars in data_iterator(filename, 
                                    E_mapping, 
                                    R_mapping, 
                                    batch_size = -1,
                                    mode = '1-to-1',
                                    passes = 1):
        Xs, Xo, Xp = true_vars
        tmp_s.extend(Xs)
        tmp_o.extend(Xo)
        tmp_p.extend(Xp)
    
    tmp_s = np.ndarray.flatten(np.asarray(tmp_s))
    tmp_o = np.ndarray.flatten(np.asarray(tmp_o))
    tmp_p = np.ndarray.flatten(np.asarray(tmp_p))
    triples = list(zip(tmp_s, tmp_o, tmp_p))
    return triples


def sort(predictions, reverse):
    predictions = [(i,v) for i,v in enumerate(predictions)]
    predictions = sorted(predictions, key=lambda x: x[1], reverse = not reverse)
    predictions = [i for i,v in predictions]
    return predictions

def in_top_k(target, sorted_predictions, reverse, k=10):
    return target in sorted_predictions[:k]

def mmr(target, sorted_predictions, reverse):
    return 1/(sorted_predictions.index(target) + 1)
    
def mr(target, sorted_predictions, reverse):
    return sorted_predictions.index(target) + 1
    

def evaluate(model, filename, E_mapping, R_mapping, reverse = True, train_triples = []):
    
    head_ignore = defaultdict(list)
    tail_ignore = defaultdict(list)
    for s,p,o in train_triples:
        tail_ignore[(s,p)].append(o)
        head_ignore[(p,o)].append(s)
        
    triples = np.squeeze(np.asarray(list(data_iterator(filename, 
                                        E_mapping, 
                                        R_mapping, 
                                        batch_size = -1,
                                        mode = '1-to-1')))).T
    N = len(E_mapping)
    
    results = defaultdict(list)
    
    for t in tqdm(triples):
        s,p,o = t
        # tail
        S,P,O = np.repeat(s,N), np.repeat(p,N), np.arange(N)
        X = np.stack([S,P,O],axis=0).T
        X = np.delete(X,tail_ignore[(s,p)], axis=0)
        pred = model.predict(X)
        sorted_predictions = sort(pred,reverse)
        for k in [1,3,10]:
            tmp = in_top_k(o, sorted_predictions, reverse = reverse, k=k)
            results[('tail',k)].append(int(tmp))
        
        results[('tail','mr')].append(mr(o,sorted_predictions,reverse))
        results[('tail','mmr')].append(mmr(o,sorted_predictions,reverse))
        
        # head
        S,P,O = np.arange(N), np.repeat(p,N), np.repeat(o,N)
        X = np.stack([S,P,O],axis=0).T
        X = np.delete(X,head_ignore[(s,p)],axis=0)
        pred = model.predict(X)
        sorted_predictions = sort(pred,reverse)
        for k in [1,3,10]:
            tmp = in_top_k(s, sorted_predictions, reverse = reverse, k=k)
            results[('head',k)].append(int(tmp))
            
        results[('head','mr')].append(mr(s,sorted_predictions,reverse))
        results[('head','mmr')].append(mmr(s,sorted_predictions,reverse))
        
    for k in results:
        results[k] = np.mean(results[k])
        
    tmp = defaultdict(list)
    
    for k in results:
        side, metric = k
        tmp[metric].append(results[k])
    
    for k in tmp:
        tmp[k] = np.mean(tmp[k])
        
    return {**results, **tmp}

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main(model, params):
    datafolder = params['d']
    training_passes = params['t']
    eval_passes = params['e']
    predict_passes = params['p']
    batch_size = params['bs']
    drop = params['drop']
    dim = params['ed']
    
    constr_dict = {'maxnorm': MaxNorm(),'unitnorm': UnitNorm(),'nonneg': NonNeg()}
    reg_dict = {'l1': l1(0.01),'l2': l2(0.01),'l1_l2': l1_l2(0.01,0.01)}
    
    train_file = datafolder + "train.txt"
    valid_file = datafolder + "valid.txt"
    test_file = datafolder + "test.txt"
    false_train_file = datafolder + "false_train.txt"
    
    E_mapping, R_mapping = mapping([train_file, valid_file, test_file, false_train_file])

    VOC_SIZE = len(list(E_mapping.keys()))
    PRED_SIZE = len(list(R_mapping.keys()))
    
    true_train = np.squeeze(np.asarray(list(data_iterator(train_file, 
                                        E_mapping, 
                                        R_mapping, 
                                        batch_size = -1,
                                        mode = params['training_mode']))))
    
    if params['reverse_labels']: #TransE
        true_train_labels = np.zeros(len(true_train.T))
    else: 
        true_train_labels = np.ones(len(true_train.T))
    
    if params['false_mode'] == 'fromfile':
        false_train = np.asarray(list(data_iterator(false_train_file, 
                                        E_mapping, 
                                        R_mapping, 
                                        batch_size = -1,
                                        mode = params['training_mode'])))
        
    else:
        s,p,o = true_train
        false_train = np.asarray(corrupt_triples(s, p, o, params['check'], params['false_mode']))
        
    if params['reverse_labels']: 
        false_train_labels = np.ones(len(false_train.T))
    else: 
        false_train_labels = np.zeros(len(false_train.T))
    
    #train_data = np.concatenate([false_train.T,true_train.T],axis=0)
    #train_labels = np.concatenate([false_train_labels.T,true_train_labels.T],axis=0)
    #train_labels = train_labels * (1 - params['ls']) + params['ls'] / 2
    
    if params['constraint']:
        const = constr_dict[params['constraint']]
    else:
        const = None
    
    if params['regularizer']:
        reg = reg_dict[params['regularizer']]
    else:
        reg = None
    
    m = model(VOC_SIZE, PRED_SIZE, dim, embeddings_regularizer=const,embeddings_constraint=reg,dropout=params['drop'])
    
    m.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['acc'])
    
    for i in range(training_passes):
        if params['false_mode'] != 'fromfile':
            s,p,o = true_train
            false_train = np.asarray(corrupt_triples(s, p, o, params['check'], params['false_mode']))
            
        tmpX = np.concatenate([false_train.T,true_train.T],axis=0)
        tmpY = np.concatenate([false_train_labels.T,true_train_labels.T],axis=0)
        tmpY = tmpY * (1 - params['ls']) + params['ls'] / 2
        
        m.fit(tmpX, tmpY, epochs=1, shuffle=True, batch_size = batch_size)
        
        try:
            if (i % eval_passes == 0 and i != 0) or (i == training_passes and eval_passes > 0):
                if params['filtered']:
                    tmp = true_train.T
                else:
                    tmp = []
                res = evaluate(m, valid_file, E_mapping, R_mapping, params['reverse_labels'], tmp)
                print(res)
        
        except ZeroDivisionError:
            pass


        if params['store']:
            store_embedding(m, E_mapping, R_mapping, datafolder)
    
    if predict_passes > 0:
        print(predict_passes)
        test = np.squeeze(np.asarray(list(data_iterator(test_file, 
                                        E_mapping, 
                                        R_mapping, 
                                        batch_size = -1,
                                        mode = params['training_mode'])))).T
        
        pred = m.predict(test)
        pred = [p[0] for p in pred]
        
        mapping_e = reverse_dict(E_mapping)
        mapping_r = reverse_dict(R_mapping)
    
        with open(params['output_file'],'w') as f:
            for t, p in zip(test, pred):
                s,r,o = t
                s,r,o = mapping_e[s],mapping_r[r],mapping_e[o]
                string = '\t'.join(map(str,[s,r,o,p])) + '\n'
                f.write(string)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='python3 main.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', nargs='?', type=str, help='model name' , default='HolE')
    parser.add_argument('datafolder', nargs='?', metavar='datafolder', type=str, help='data set folder' , default='./data/WN18RR/')
    parser.add_argument('-t', type=int, help='train mode data passes', default=0)
    parser.add_argument('-e', type=int, help='evaluation for ever N training passes', default=0)
    parser.add_argument('-p', type=int, help='predict mode data passes', default=0)
    parser.add_argument('-ed', type=int, help='Embedding dim', default=128)
    parser.add_argument('-bs', type=int, help='Batch size', default=2048)
    parser.add_argument('-lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('-ls', type=float, help='Lable smoothing', default=0.0)
    parser.add_argument('-drop', type=float, help='Dropout', default=0.2)
    parser.add_argument('-width', type=float, help='For ConvE, width of "image"', default=8)
    parser.add_argument('-height', type=float, help='For ConvE, height of "image"', default=8)
    parser.add_argument('-constraint', help='Embedding constraint (maxnorm, unitnorm, nonneg).', type=str, default=None)
    parser.add_argument('-regularizer', help='Embedding regularizer (l1,l2,l1_l2).', type=str, default=None)
    parser.add_argument('-threshold', type=float, help='stop training if MRR change is less than threshold (use in conjunction with evaluation)', default = 0.0)
    
    
    parser.add_argument('--store', help='Store embedding', action='store_true')
    parser.add_argument('--filtered', help='Filter results', action='store_true')
    parser.add_argument('--check', help='Check false triples against existing in KG.', action='store_true')
    parser.add_argument('model_dir', nargs='?', metavar='model_dir', type=str, help='name of directory to save model.' , default='./saved_model/test')
    parser.add_argument('false_mode', nargs='?', metavar='false_mode', type=str, help='false triple mode (random, domain, range, compliment_domain, compliment_range, semantic, fromfile)' , default='random')
    parser.add_argument('training_mode', nargs='?', metavar='training_mode', type=str, help='1-to-1, N-to-1, 1-to-N, or N-to-N' , default='1-to-1')
    parser.add_argument('output_file', nargs='?', metavar='output_file', type=str, help='where to save performance metrics.' , default='output.txt')
    parser.add_argument('eval_file', nargs='?', metavar='eval_file', type=str, help='where to save evaluation metrics.' , default='eval.txt')
    

    args = parser.parse_args()
    
    params = {}
    
    params['t'] = args.t
    params['e'] = args.e
    params['p'] = args.p
    params['d'] = args.datafolder
    params['ed'] = args.ed
    params['bs'] = args.bs
    params['lr'] = args.lr
    #params['margin'] = args.ma
    #params['margin_increase'] = args.mi
    params['w'] = args.width
    params['h'] = args.height
    #params['mmax'] = args.mmax
    params['drop'] = args.drop
    params['store'] = args.store
    params['filtered'] = args.filtered
    params['check'] = args.check
    #params['explain'] = args.explain
    params['model_dir'] = args.model_dir
    params['output_file'] = args.output_file
    params['eval_file'] = args.eval_file
    params['false_mode'] = args.false_mode
    params['threshold'] = args.threshold
    params['training_mode'] = args.training_mode
    params['constraint'] = args.constraint
    params['regularizer'] = args.regularizer
    params['ls'] = args.ls
    
    params['reverse_labels'] = False
 
    if args.model == 'HolE':
        from models import HolE as model
    elif args.model == 'ConvE':
        from models import ConvE as model
    elif args.model == 'TransE':
        params['reverse_labels'] = True
        from models import TransE as model
    elif args.model == 'DistMult':
        from models import DistMult as model
    elif args.model == 'ComplEx':
        from models import ComplEx as model
    else:
        raise NotImplementedError (args.model)
  
    main(model = model, params = params)
    
