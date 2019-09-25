"""
false_triples.py

Creates false training triples under CWA. 
"""

from random import choice
from collections import defaultdict
import numpy as np
from false_generator import FalseGenerator 

def get_domains(subjects, predicates):
    out = defaultdict(set)
    for s,p in zip(subjects,predicates):
        out[p].add(s)
    return out

def remove_existing(triples, all_triples):
    # Remove all_triples from triples

    triples = set(map(tuple, triples))
    all_triples = set(map(tuple, all_triples))
    res = triples - all_triples
    
    return list(res)

def inverse_domain(domains, all_entities):
    for k,i in domains.items():
        domains[k] = set(all_entities) - set(i)
        if len(domains[k]) < 1:
            domains[k] = set(all_entities)
    return domains

def get_false_triple(subjects, objects, predicate):
    return (choice(subjects), choice(objects), predicate)

def corrupt_triples(true_subjects, true_predicates, true_objects, check = False, mode = 'random', false_gen = None):
    """
    Create false triples.
    args:
        true_subjects :: list 
                all subjects
        true_objects :: list 
                all objects
        true_predicates :: list 
                all predicates
        check :: bool
                remove existing true triples from created false triples.
        mode :: string
                random :: assume closed world, perturb subject/object with randomly selected from set of all entities.
                domain :: keep domain and range consistent across false and true triples.
                range :: reverse domain and range for relation.
                compliment_domain :: use all entities minus the true domain/range as domain/range.
                compliment_range :: use all entities minus the true domain/range as range/domain.
                semantic :: use class assertions, domain/range statements etc. to construct open world false triples. 
        ontology :: Not implemented.
                required by mode = 'semantic'
    """
    true_subjects = list(np.ndarray.flatten(true_subjects))
    true_objects = list(np.ndarray.flatten(true_objects))
    true_predicates = list(np.ndarray.flatten(true_predicates))
    all_entities = list(set(true_subjects).union(set(true_objects)))
    
    false_triples = []
    
    if mode == 'random':
        for p in true_predicates:
            false_triples.append(get_false_triple(all_entities, all_entities, p))
            
    elif mode == 'domain':
        # Domains and ranges used with predicate in KF.
        domains = get_domains(true_subjects, true_predicates)
        ranges = get_domains(true_objects, true_predicates)
        for p in true_predicates:
            false_triples.append(get_false_triple(list(domains[p]),list(ranges[p]),p))
            
    elif mode == 'range':
        # Domains and ranges used with predicate in KF.
        domains = get_domains(true_objects, true_predicates)
        ranges = get_domains(true_subjects, true_predicates)
        for p in true_predicates:
            false_triples.append(get_false_triple(list(domains[p]),list(ranges[p]),p))
            
    elif mode == 'compliment_domain':
        # newD = Compliment(D),newR = Compliment(R)
        domains = get_domains(true_subjects, true_predicates)
        ranges = get_domains(true_objects, true_predicates)
        domains = inverse_domain(domains, all_entities)
        ranges = inverse_domain(ranges, all_entities)
        for p in true_predicates:
            false_triples.append(get_false_triple(list(domains[p]),list(ranges[p]),p))
    
    elif mode == 'compliment_range':
        # newD = Compliment(R), newR = Compliment(D)
        domains = get_domains(true_objects, true_predicates)
        ranges = get_domains(true_subjects, true_predicates)
        domains = inverse_domain(domains, all_entities)
        ranges = inverse_domain(ranges, all_entities)
        for p in true_predicates:
            false_triples.append(get_false_triple(list(domains[p]),list(ranges[p]),p))
    elif mode == 'ontology':
        if not isinstance(false_gen, FalseGenerator):
            raise TypeError(mode, 'requires a FalseGenerator object as input.')
        
        for s,p,o in zip(true_subjects,true_predicates,true_objects):
            method = choice(['range','domain','disjoint'])
            false_triples.append(false_gen.corrupt((s,p,o),method=method))
        
    else:
        raise NotImplementedError(mode + " not implemented")
        
            
    # Check if false triples already exists in KG.
    if check:
        false_triples = remove_existing(false_triples, list(zip(true_subjects,true_objects,true_predicates)))
    
    # Extend to correct size and save.
    while len(true_subjects) > len(false_triples):
        false_triples.extend(false_triples)
    false_triples = false_triples[:len(true_subjects)]
    
    false_subjects, false_objects, false_predicates = zip(*false_triples)
    
    return false_subjects, false_predicates, false_objects

