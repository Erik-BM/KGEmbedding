# KG Embeddings

## Terminology
* **Embedding:** A mapping from a entities or relation to a vector in the embedding space.
* **Scoring function:** Gives a score to a triple, the meaning of the score depends on the model. e.g. TransE optimizes for a low score for true triples while HolE optimizes for a score as close as possible to 1 for true triples and 0 for false triples.
* **Knowledge graph:** In this context, a KG is simply a set of triples.
* **True triple:** A triple in the KG.
* **False/corrupt triple:** A triple that is not in the KG.

## Introduction to knowledge graph embedding
The aim of knowledge graphs is to gather information in an easy reachable format. But, most KGs are not complete, this is where link prediction comes in. The goal is to assign a score/probability to new triples and then add them to the KG if the score is higher that some threshold.

To perform link prediction on a KG we first need a representation of it in a vector space. This is achieved by embedding, a context/semantic aware dimension reduction (analog to NLP embedding, e.g. Word2Vec). The target for a good embedding is to preserve the relationships between entities that are in the KG, while also position entities in the vector space such that we can infer new relationships i.e. do link predication. 

The preserving of the relationships varies from model to model. The naive way is to use distance in the vector space as a measure. e.g. Embedding of England, London, Spain and Madrid should achieve similar distance between England and London as for Spain and Madrid. This since they have a relationship in common, namely capital. The TransE model uses this method exactly.

To train a model, we solve an optimization problem, usually a minimax problem. Maximizing the score of true triple while minimizing the score of corrupted triples. The corrupt triples are in most cases a perturbed version of a true triple, swapping out either subject, predicate or object, or a combination for random entities/relation in the KG.

Recommended reads:
* [Bordes et. al.](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

## Models
Below are a selection of the most common models used in KG embedding. More models will be added in the future.

### HolE
Holographic embeddning use the fact that correlation is not a commutative property to model directional relationships better than models with commutative scoring function.
The scoring function for a triple (subject, object, predicate) is `E(s,o,p) = f(p^T (K(s,o)))`, where f is an activation function (sigmoid), s, o, p are embedded vectors and `K(s,o) =  ifft(conj(fft(s)) * fft(o))` (`fft` and `ifft` are the fast Fourier transforms).

For each triple T in the training set we create a corrupt triple C which is not in the KG. Then we are left with a minimax problem. 

### ConvE
Convolutional embeddings uses a 2D convolution neural network layer to learn embeddings. This approach is very parameter efficient, achieving similar results as other models with only the fraction of the parameters.

### TransE
TransE uses a simple and intuitive scoring function, namely that the embedded object of a subject-predicate pair should be close to the sum of the subject and predicate embedding. `E(s,o,p) = dist(s + p, o)`.

### DistMult
DistMult uses the scoring function `E(s,o,p) = sum_i (s_i * o_i * p_i)` or a triple inner product. This leads to a high score for triples where the elements of `s`, `o`, and `p` are similar (in sign, either same or alternating) and large. We normalize the score with a sigmoid activation function. 

### ComplEx
Basically, ComplEx is the same as DistMult in that it has the same scoring function. However, the embeddings are complex and the scoring function takes the conjugate of `p`. i.e. `E(s,o,p) = sum_i (s_i * o_i * p_i^*) = sum_i[(Re(s_i) * Re(o_i) * Re(p_i)) + (Re(s_i) * Im(o_i) * Im(p_i)) + (Re(s_i) * Im(o_i) * Im(p_i)) - (Im(s_i) * Re(o_i) * Im(p_i))]`. This results in that ComplEx is able to learn symmetric and antisymmetric relations.

By default, binary cross-entropy is used as a loss function. However, all losses defined by [keras](https://keras.io/losses/) can be used.


### Evaluation
There are (mainly) two methods for evaluating the predictive performance of a model. These are Hits@k and mean reciprocal rank (MRR). Hits@k is true if the true triple are within the k larges scores for all possible objects or subjects. We use the average of testing all subjects and all objects. 
MRR is the mean of the sum of the inverse of the rank. This means that the metric does not have a cut off at top k, but rather get a contribution from all predictions.

## Prerequisites
Tested on Ubuntu 16.04 LTS with NVIDIA GTX 1050 and Ubuntu 18.04 LTS on CPU. All tests with Python 3.6.5. Under training memory restrictions is a non-issue, more than 2GB is sufficient.

Install the required packages.
```
pip3 install -r requirements.txt
```
Click [here](https://www.tensorflow.org/install/) for TensorFlow install instructions.

Finally, create two folders for storing the models and temporary files.
```
mkdir tmp saved_models
```

## File descriptions
### main.py
Set up of estimators. Performs training and evaluations.
### common/DataPrep.py
Prepare data for use by the estimator. Create training data generator for the number of training iterations specified. Other helping function, such as a priori probability calculator.
### common/FalseTriples.py
Create false triples from given true triples. Modes for random, domain/range, and semantic construction of triples.
### DataPrep.py and CreateMapping.py
Converting RDF file containing triples to data for the models. Adjustable train/eval/test proportions.

## Example
Run
```
python3 main.py -h
```
for help.

Training HolE over WN18 for 100 epochs with evaluation every 10 epochs:
```
python3 main.py HolE WN18 -t 100 -e 10
```

## Parameters

List of datasets:
```
FB15k-237
kinship
nations
UMLS
WN18
WN18RR (WN18 with reverse relation leaking removed.)
YAGO3-10
```
List of models:
```
HolE
ConvE
TransE
DistMult
ComplEx
```
It is also possible to specify which device the program should run on, write 
```
CUDA_VISIBLE_DEVICES="-1" python3 main.py ......
```
for CPU only. Replace `"-1"` to specify which GPU to use. If no environment variable is set, the program will use the default for TensorFlow.


## Authors

* **Erik Bryhn Myklebust** - *Initial work* - [Erik-BM](https://gitlab.com/Erik-BM)


## Bibliography
* **Holographic Embeddings of Knowledge Graphs** (AAAI-16)
  by M. Nickel et. al.__
  ([pdf](https://arxiv.org/pdf/1510.04935.pdf))
  
* **Convolutional 2D Knowledge Graph Embeddings** (AAAI-18)
  by T. Dettmers et. al.  
  ([pdf](https://arxiv.org/pdf/1707.01476.pdf))
  
* **Translating Embeddings for Modeling Multi-relational Data** 2013
  by Bordes et. al.
  ([pdf](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf))
  
* **Embedding Entities and Relations for Learning and Inference in Knowledge Bases** 2015
  by Yang et. al.
  ([pdf](https://arxiv.org/pdf/1412.6575.pdf))
  
* **Complex Embeddings for Simple Link Prediction** 2016
  by Trouillon et. al.
  ([pdf](https://arxiv.org/pdf/1606.06357.pdf))
