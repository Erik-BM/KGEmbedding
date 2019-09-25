"""
Convert RDF to files for embedding training.

Will create:
train.txt, training triples separated by line skip. subject,predicate,object
test.txt, evaluation triples
test.txt, testing triples

"""

import rdflib
import argparse
from random import choice
import string
import numpy as np
import math
import os

def divide_list(original_list, weight_list):
    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil( (len(original_list) * weight) )

        sublists.append( original_list[prev_index : next_index] )
        prev_index = next_index

    return sublists

def id_generator(size=20, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(choice(chars) for _ in range(size))

def save_mapping(mapping, filename):
    with open(filename, 'w') as f:
        for k,i in mapping.items():
            f.write(k + '\t' + i + '\n')

def save_triples(triples, entitiy_mapping, relation_mapping, filename):
    with open(filename, 'w') as f:
        for s,p,o in triples:
            tmp = entitiy_mapping[s] +'\t'+ relation_mapping[p] +'\t'+ entitiy_mapping[o] + '\n'
            f.write(tmp)

def load_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            k,i = l.split()
            mapping[k] = i
    return mapping

def main(params):
    g = rdflib.Graph()
    g.parse(params['filename'], format = params['extension'])
    
    idx = np.random.permutation(len(g))
    
    train_idx, eval_idx, test_idx = divide_list(idx, [params['train'], params['eval'], params['test']]) 
    
    train_triples = []
    eval_triples = []
    test_triples = []
    literal_triples = []
    data_properties = set()
    
    c = 0
    for s,p,o in g:
        if any([rdflib.URIRef(p) == a for a in params['ignore']]):
            continue
        if isinstance(o, rdflib.Literal):
            literal_triples.append((s,p,o))
            c += 1
            continue
        s = str(s)
        p = str(p)
        o = str(o)
        if c in train_idx:
            train_triples.append((s,p,o))
        if c in eval_idx:
            eval_triples.append((s,p,o))
        if c in test_idx:
            test_triples.append((s,p,o))
        c += 1
    
    entitiy_mapping = load_mapping(params['entitiy_mapping'])
    relation_mapping = load_mapping(params['relation_mapping'])
    data_relation_mapping = load_mapping(params['data_relation_mapping'])
    
    directory = os.path.splitext(params['filename'])[0]
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    save_mapping(entitiy_mapping, directory + '/entities.txt')
    save_mapping(relation_mapping, directory + '/relations.txt')
    save_mapping(data_relation_mapping, directory + '/data_relations.txt')
    
    save_triples(train_triples, entitiy_mapping, relation_mapping, directory + '/train.txt')
    save_triples(literal_triples, entitiy_mapping, data_relation_mapping, directory + '/literal.txt')
    save_triples(eval_triples, entitiy_mapping, relation_mapping, directory + '/valid.txt')
    save_triples(test_triples, entitiy_mapping, relation_mapping, directory + '/test.txt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 CreateData.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', type=str, help='file containing triples')
    parser.add_argument('--eval', nargs='?', type=float, help='evaluation set fraction', default = 0.1)
    parser.add_argument('--test', nargs='?', type=float, help='test set fraction', default = 0.1)
    parser.add_argument('relation_mapping', type=str, help='filename, relation mapping, created with CreateMapping.py')
    parser.add_argument('entitiy_mapping', type=str)
    parser.add_argument('data_relation_mapping', type=str)
    parser.add_argument('ignore', nargs='?', type=str, help='Relations to ignore. Write all relations as string separated by a space.')
    
    args = parser.parse_args()
    params = {}
    params['filename'] = args.filename
    params['extension'] = args.filename.split('.')[-1]
    params['eval'] = args.eval
    params['test'] = args.test
    params['train'] = 1 - args.eval - args.test
    params['entitiy_mapping'] = args.entitiy_mapping
    params['relation_mapping'] = args.relation_mapping
    params['data_relation_mapping'] = args.data_relation_mapping
    try:
        params['ignore'] = [rdflib.URIRef(a) for a in args.ignore.split(' ')]
    except AttributeError:
        params['ignore'] = []
    
    main(params)
    
