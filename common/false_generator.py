"""
GOAL:
Create incositent triples. 

version 1

- RDFS.domain
- RDFS.range
- OWL.disjointWith
- OWL.equivalentClasses


max and min cardinality makes no sense to use. Voiding max will make more true triples false, while min requires removing triples from Abox. 

"""

from rdflib import Graph, Literal, BNode
from rdflib.namespace import RDFS, OWL, RDF
from collections import defaultdict
import warnings

from random import choice

class FalseGenerator:
    def __init__(self, tboxfile, aboxfile):
        """
        Create inconsitent triples based on an ontology. 
        
        args
            tboxfile :: string
                path to TBox file, rdf/owl format
            aboxfile :: string 
                path to ABox file, any format supported by rdflib
        """
        
        self.ontology = Graph()
        self.ontology.parse(tboxfile)

        self.data = Graph()
        f = aboxfile.split('.')[-1]
        self.data.parse(aboxfile,format=f)
        
        self._perform_classification()
        
        self.entities = set(self.data.subjects()) | set(self.data.objects())
    
        self._load_false_domains()
        self._load_false_ranges()
        self._load_disjoint()
    
    def _perform_classification(self):
        l1 = len(self.data)
        
        while True: #crude method, but works
            for child, parent in self.ontology.subject_objects(predicate=RDFS.subClassOf):
                for s in self.data.subjects(predicate=RDF.type, object=child):
                    self.data.add((s,RDF.type,parent))
                
            l2 = len(self.data)
            if not l1 < l2:
                break
            l1 = l2
    
    def _load_false_domains(self):
        self.D = defaultdict(set)
        for p,c in self.ontology.subject_objects(RDFS.domain):
            self.D[p] |= set(self.data.subjects(predicate = RDF.type, object = c))
        
    def _load_false_ranges(self):
        self.R = defaultdict(set)
        for p,c in self.ontology.subject_objects(RDFS.range):
            self.R[p] |= set(self.data.subjects(predicate = RDF.type, object = c))
            
    def _load_disjoint(self):
        self.disjoint = defaultdict(set)
        for p,c in self.ontology.subject_objects(OWL.disjointWith):
            self.disjoint[p].add(c)
        
        for p,c in self.ontology.subject_objects(OWL.equivalentClass):
            self.disjoint[p] |= self.disjoint[c]
            
    def _corrupt_entity(self, p, e, method = 'disjoint'):
        
        if method == 'disjoint':
            """
            Find classes disjoint to C(e). The choose a random entity from the union of these classes.
            """
            cls = self.data.objects(subject=e, predicate=RDF.type)
            disjoint_entities = set()
            for c in cls:
                disjoint_entities |= set.union(*[self.data.subjects(predicate=RDF.type, object=dc) for dc in self.disjoint[c]])
                    
            if disjoint_entities:
                e = choice(list(disjoint_entities))
            else:
                e = None
            
        elif method == 'range':
            try:
                e = choice(list(self.entities.difference(self.R[p])))
            except KeyError:
                e = None
            
        elif method == 'domain':
            try:
                e = choice(list(self.entities.difference(self.D[p])))
            except KeyError:
                e = None
        
        elif method == 'random':
            e = choice(list(self.entities))
            
        else:
            raise NotImplementedError(method, 'not implemented.')
        
        return e
    
    def corrupt(self, t = None, method='range', ignore_literals=True, ignore_blank_nodes = True):
        """
        Corrupt a input triple to make it incositent with TBox.
        
        Args:
            t :: tuple 
                a triple s,p,o
            method :: string or list
                methods to use for corrupting triples
                
                method = 'range' :: a new object is selected from the complement of the predicates range. 
                method = 'domain' :: a new subject is selected from the complement of the predicates domain. 
                method = 'disjoint' :: in conjunction with domain or range. a new subject/object is selected based on disjoint classes from input subject/object.
                
            ignore_literals :: bool
                corrupt with literals 
            ignore_blank_nodes :: bool
                corrupt with blank nodes 
        
        Returns:
            corrupt triple (tuple)
        """
        
        if not isinstance(method, list):
            method = [method]
        
        if t:
            s,p,o = t
        else:
            s,p,o = choice(list(self.data))
            if ignore_blank_nodes and (isinstance(s,BNode) or isinstance(o,BNode)):
                return self.corrupt(None, method, ignore_literals, ignore_blank_nodes)
            
            if ignore_literals and (isinstance(s,Literal) or isinstance(o,Literal)):
                return self.corrupt(None, method, ignore_literals, ignore_blank_nodes)
        
        if 'domain' in method:
            s = self._corrupt_entity(p,s,'domain')
            
            if isinstance(s, Literal): #subject cannot be literal
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
            if ignore_blank_nodes and isinstance(s,BNode):
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
        
        if 'range' in method:
            o = self._corrupt_entity(p,o,'range')
            
            if ignore_literals and isinstance(o,Literal):
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
            if ignore_blank_nodes and isinstance(o,BNode):
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
     
        if 'disjoint' in method:
            if 'range' in method:
                o = self._corrupt_entity(p,o,'disjoint')
            if 'domain' in method:
                s = self._corrupt_entity(p,s,'disjoint')
            
            if (ignore_literals and isinstance(o,Literal)) or isinstance(s,Literal):
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
            if ignore_blank_nodes and (isinstance(o,BNode) or isinstance(s,BNode)):
                return self.corrupt((s,p,o), method, ignore_literals, ignore_blank_nodes)
        
        if 'random' in method:
            s = choice(list(self.entities))
            o = choice(list(self.entities))
           
        if not all([s,p,o]):
            warnings.warn('No corrupt entity found: returning random triple')
            return self.corrupt(None, ['random'], ignore_literals, ignore_blank_nodes)
        
        return s,p,o
