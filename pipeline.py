import os
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree
from nltk.treetransforms import chomsky_normal_form as cnf

import re
from collections import Counter

import numpy as np
import torch
import nltk
import turtle

import dgl


class Pipeline:
        
    def __init__(self, path, vocab_size, parser):
        self.MAX_VOCAB_SIZE = vocab_size
        self.path = path
        self.corpus = self.__build_corpus(path)
        self.idx2Word = self.corpus[0]
        self.word2Idx = self.corpus[1]
        self.parser = parser
    
    def text_cleaner(self, text):
        # lower case text
        newString = text.lower()
        newString = re.sub(r"'s\b","",newString)
        # remove punctuations
        newString = re.sub("[^a-zA-Z]", " ", newString) 
        #long_words=[]
        # remove short word
        #for i in newString.split():
        #    if len(i)>=3:                  
        #        long_words.append(i)
        #return (" ".join(long_words)).strip()
        return newString
    
    
    def __build_corpus(self, path):
        with open(path) as fin:
            text = fin.read()
    
        # preprocess the text
        text = self.text_cleaner(text).split()

        vocab = dict(Counter(text).most_common(self.MAX_VOCAB_SIZE-1))
        vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

        idx2Word = [word for word in vocab.keys()] # index to word (referrence from integer to string) 
        word2Idx = {word:i for i, word in enumerate(idx2Word)} # word to index (referrence from string to integer)

        return idx2Word, word2Idx
    
    
    # define string to integer transformation
    def str2Int(self, word):
        if word.isupper():
            # PAD_WORD
            return -1
        else:
            # normal word, look up for its index in the dictionary
            return self.word2Idx[word]
        
        
    # define integer to string transformation
    def int2Str(self, idx):
        return self.idx2Word[idx]
    
    def masker(self, value):
        if value==-1:
            return 0
        else:
            return 1
    
    def text2DGL(self, sent):
        clean_sent = self.text_cleaner(sent)
        lt = self.parser.raw_parse(clean_sent)
        cc_tree = next(lt)
        
        node_out, node_in, ndata_x, ndata_y, ndata_mask = self.topology(cc_tree)
        g = dgl.graph((node_out, node_in))
        g.ndata['x'] = torch.tensor(ndata_x)
        return g
        
    
    
    def topology(self, tree):
        sudo_idx_data_dict = {}
        idx_data_dict = {}
        sudo_idx_real_idx_dict = {}
        out_edge_list = []
        in_edge_list = []
        pt_tree = ParentedTree.convert(tree)
        idx = 0

        for subtree in pt_tree.subtrees():
            if subtree.height() == 2:
                child_sudo_idx = subtree.treeposition()
                child_content = subtree.flatten()[0]
                parent_sudo_idx = subtree.parent().treeposition()
                parent_content = subtree.parent().label()

                out_edge_list.append(child_sudo_idx)
                in_edge_list.append(parent_sudo_idx)
                sudo_idx_data_dict[child_sudo_idx] = self.str2Int(child_content)
                sudo_idx_data_dict[parent_sudo_idx] = self.str2Int(parent_content)

                #print('child_sudo_idx:', child_sudo_idx, 'child_content:', child_content)
                #print('parent_sudo_idx:', parent_sudo_idx, 'parent_content:', parent_content)

            else:
                if subtree.parent() is None:
                    print('Start building:\n', '\t', ' '.join(tree.flatten()))
                else:
                    child_sudo_idx = subtree.treeposition()
                    child_content = subtree.label()
                    parent_sudo_idx = subtree.parent().treeposition()
                    parent_content = subtree.parent().label()

                    out_edge_list.append(child_sudo_idx)
                    in_edge_list.append(parent_sudo_idx)
                    sudo_idx_data_dict[child_sudo_idx] = self.str2Int(child_content)
                    sudo_idx_data_dict[parent_sudo_idx] = self.str2Int(parent_content)

                    #print('child_sudo_idx:', child_sudo_idx, 'child_content:', child_content)
                    #print('parent_sudo_idx:', parent_sudo_idx, 'parent_content:', parent_content)

        ls = list(sudo_idx_data_dict.items())
        for real_idx in range(len(ls)):
            (sudo_idx, _) = ls[real_idx]
            sudo_idx_real_idx_dict[sudo_idx] = real_idx

        #print(sudo_idx_real_idx_dict)

        for counter in range(len(out_edge_list)):
            sudo_idx_out = out_edge_list[counter]
            sudo_idx_in = in_edge_list[counter]
            out_edge_list[counter] = sudo_idx_real_idx_dict[sudo_idx_out]
            in_edge_list[counter] = sudo_idx_real_idx_dict[sudo_idx_in]

        #print(out_edge_list, in_edge_list)

        for (k, v) in sudo_idx_data_dict.items():
            # k = sudo_idx_real_idx_dict[k]
            idx_data_dict[sudo_idx_real_idx_dict[k]] = sudo_idx_data_dict[k]

        #print(idx_data_dict)

        _, value_list = zip(*(sorted(list(idx_data_dict.items()))))
                
        return out_edge_list, in_edge_list, list(value_list), list(label_list), mask_list
    
    
    def draw_graph(self, sent):
        sent = self.text_cleaner(sent)
        lt = self.parser.raw_parse(sent)
        cc_tree = next(lt)

        # constitucency tree
        nltk.tree.Tree.draw(cc_tree)
        return cc_tree
