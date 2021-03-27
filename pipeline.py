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
