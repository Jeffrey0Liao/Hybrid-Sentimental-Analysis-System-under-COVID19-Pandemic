import os
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree
from nltk.treetransforms import chomsky_normal_form as cnf
from nltk.sentiment import SentimentIntensityAnalyzer

import re
from collections import Counter

import numpy as np
import torch
import nltk
import turtle

from transformers import pipeline

import dgl

# Raw text processing pipeline

# hyperparameters
MAX_VOCAB_SIZE = 1000
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

# build language model and dictionary

# build generic corpus as dictionaries to achieve bidirectional referrence

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()
  
# model path for standford parser
path_to_jar='C:\\Users\\41713\\FYP\\stanford-parser-full-2015-12-09\\stanford-parser.jar'
path_to_models_jar='C:\\Users\\41713\\FYP\\stanford-parser-full-2015-12-09\\stanford-parser-3.6.0-models.jar'
the_model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"

# system environment
os.environ['STANFORD_PARSER'] = path_to_jar
os.environ['STANFORD_MODELS'] = path_to_models_jar

# natural language parser (Stanford parser utilized)

parser = StanfordParser(model_path=the_model_path) # identify parser
print(parser)

sent = text_cleaner(sentence)
lt = parser.raw_parse(sent)
cc_tree = next(lt)

# constitucency tree
print(type(cc_tree))
print(cc_tree)
nltk.tree.Tree.draw(cc_tree)

# dtype conversion
# specify and copy graph topology
# from NLTK format to generic handlable format for pytorch or DGL dataset

node_idx_dict = {}
node_value_dict = {} 

# define string to integer transformation
def str2Int(word):
    if word.isupper():
        # PAD_WORD
        return -1
    else:
        # normal word, look up for its index in the dictionary
        return word2Idx[word]

# define node number to construct graph
def node2Idx(node):
    (idx, label) = node
    if idx in node_idx_dict.keys():
        return node_idx_dict[idx]
    else:
        node_idx_dict[idx] = label
        return idx

#
def node2Value(node):
    (idx, label) = node
    if idx in node_value_dict.keys():
        return node_value_dict[idx]
    else:
        value = str2Int(label)
        node_value_dict[idx] = value
        return value
    
# define traversal
def tree_topology(tree):
    out_edge_list = []
    in_edge_list = []
    pt_tree = ParentedTree.convert(tree)
    idx = 0
    
    for subtree in pt_tree.subtrees(): 
        child = (idx, subtree.label())
        idx += 1       
        print('child:',node_value(child),'child node:',node_num(child))        
        
        if subtree.parent() is None:
            print('parent:','NONE','parent node:','NONE')
        else:
            parent = (idx, subtree.parent().label())
            idx += 1
            print('parent:',node_value(parent),'parent node:',node_num(parent))
            
            out_edge_list.append(node_num(child))
            in_edge_list.append(node_num(parent))
        print('-----------done----------')
        
        if subtree.height()==2:
            
            parent = (idx, subtree.label())
            idx += 1
            print('child:',node_value(child),'child node:',node_num(child))
            print('parent:',node_value(parent),'parent node:',node_num(parent))
            print('----------done-----------')
            
            out_edge_list.append(node_num(subtree.flatten()[0]))
            in_edge_list.append(node_num(subtree.label()))

        
    key_list, value_list = zip(*(sorted(list(node_value_dict.items()))))
    return out_edge_list, in_edge_list, list(value_list)

