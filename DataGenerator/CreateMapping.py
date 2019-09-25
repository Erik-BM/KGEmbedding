"""
Create mapping for all entities and relations in the data set.

Will create:
entities.txt, all entities on form (entity, random string)
relations.txt, all relations

All entities and relations are exchanged for random strings when creating the data files. 

"""

import rdflib
import argparse
from random import choice
import string
import numpy as np
import math
import os
from os.path import basename

def split(original_list, weight_list):
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
            f.write(k + ' ' + i + '\n')

def save_triples(triples, entitiy_mapping, relation_mapping, filename):
    with open(filename, 'w') as f:
        for s,p,o in triples:
            tmp = entitiy_mapping[s] +' '+ relation_mapping[p] +' '+ entitiy_mapping[o] + '\n'
            f.write(tmp)

def load_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            k,i = l
            mapping[k] = i
    return mapping

def main(params):
    g = rdflib.Graph()
    g.parse(params['filename'], format = params['extension'])
    
    all_entities = set()
    all_relations = set()
    all_data_relations = set()
    
    for s,p,o in g:
        if any([rdflib.URIRef(p) == a for a in params['ignore']]):
            continue
        if isinstance(o, rdflib.Literal):
            p = str(p)
            all_data_relations.add(p)
        s = str(s)
        o = str(o)
        p = str(p)
        all_entities.add(s)
        all_entities.add(o)
        all_relations.add(p)
    
    directory = os.path.splitext(params['filename'])[0]
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    ids = [id_generator() for _ in range(len(all_data_relations) + len(all_entities) + len(all_relations))]
    i = 0
    entitiy_mapping = {}
    for e in all_entities:
        entitiy_mapping[e] = ids[i]
        i += 1
    relation_mapping = {}
    for e in all_relations:
        relation_mapping[e] = ids[i]
        i += 1
    data_relation_mapping = {}
    for e in all_data_relations:
        data_relation_mapping[e] = ids[i]
        i += 1
    
    save_mapping(entitiy_mapping, directory + '/entities.txt')
    save_mapping(relation_mapping, directory + '/relations.txt')
    save_mapping(data_relation_mapping, directory + '/data_relations.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 CreateMapping.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', nargs='?', type=str, help='filename')
    parser.add_argument('ignore', type=str, help='Relations to ignore. In string separated by space.')
    
    args = parser.parse_args()
    params = {}
    params['filename'] = args.filename
    params['extension'] = args.filename.split('.')[-1]
    try:
        params['ignore'] = [rdflib.URIRef(a) for a in args.ignore.split(' ')]
    except AttributeError:
        params['ignore'] = []
    
    main(params)
    
