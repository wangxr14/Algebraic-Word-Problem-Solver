import sys
import os
import math
from util import *
from nltk.parse import stanford
from nltk.tree import ParentedTree


def find_associated_verb(node):
    a_verb_node = None
    verb_labels=['VB','VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    vp_labels=['VP']
    
    while node:
        if node.label() in verb_labels:
            a_verb_node = node
            return a_verb_node
        elif node.label() in vp_labels:
            for i in node:
                if i.label() in verb_labels:
                    a_verb_node = i
                    return a_verb_node
        else:
            # Check siblings
            tmp=node.left_sibling()
            while tmp:
                if tmp.label() in verb_labels:
                    a_verb_node = tmp
                    return a_verb_node
                elif tmp.label() in vp_labels:
                    for i in tmp:
                        if i.label() in verb_labels:
                            a_verb_node = i
                            return a_verb_node
                tmp=tmp.left_sibling()
            
            tmp=node.right_sibling()
            while tmp:
                if tmp.label() in verb_labels:
                    a_verb_node = tmp
                    return a_verb_node
                elif tmp.label() in vp_labels:
                    for i in tmp:
                        if i.label() in verb_labels:
                            a_verb_node = i
                            return a_verb_node
                tmp=tmp.right_sibling()
        node = node.parent()
    return a_verb_node

def find_subject(node):
    subject_node = None
    while node:
        if node.left_sibling():
            subject_node = node.left_sibling()
            return subject_node
        node=node.parent()
    return subject_node    

def find_unit(node):
    unit_node = None
    phrase_labels=['VP','NP','PP']
    while node:
        if node.label() in phrase_labels:
            unit_node = node
            break
        else:
            node = node.parent()
    return unit_node
    
def find_np(unit_node):
    r_np_node=[]
    while unit_node:
        if unit_node.right_sibling() and unit_node.right_sibling().label() == 'PP':
            r_np_node.append(unit_node.right_sibling())
            node = unit_node.right_sibling()
            if node.right_sibling() and node.right_sibling().label() == 'NP':
                r_np_node.append(node.right_sibling())
                unit_node = node.right_sibling()
            else:
                break
        else:
            break
    return r_np_node

def find_rate(unit_node):
    rate_words=['per', 'each']
    rate=[]
    leaves=unit_node.leaves()
    for leaf in leaves:
        if leaf in rate_words:
            for l in leaves:
                if not l==leaf:
                    rate.append(l)
    return rate
    
def dealWithSentence(root):
    retObj={}
    # Get quantities
    retObj['quantities']=[]
    quantity_list=[]
    quantity_label='CD'
    def traverse(t):
        if t.height()==2:
            if t.label()==quantity_label:
                quantity_list.append(t)
                retObj['quantities'].append(t.leaves()[0])
            return
        else:
            for child in t:
                traverse(child)
    traverse(root)
    
    count = 0
    for q in quantity_list:
        quantityObj = {}
        node = q.parent()
        # Detect associated verb
        a_verb_node = find_associated_verb(node)
        quantityObj['verb']=a_verb_node.leaves()[0]
        
        # Subject
        subject_node = find_subject(a_verb_node)
        if subject_node:
            quantityObj['subject']=' '.join(subject_node.leaves())
        else:
            quantityObj['subject']=''
        
        # Unit
        unit_node = find_unit(q.parent())
        units=[]
        for i in unit_node:
            if i.label()==quantity_label:
                continue
            else:
                for item in i.leaves():
                    units.append(item)
        # Followed by PP-NP
        if unit_node.right_sibling() and unit_node.right_sibling().label()=='PP':
            i = unit_node.right_sibling()
            for item in i.leaves():
                    units.append(item)
            if i.right_sibling() and i.right_sibling().label()=='NP':
                for item in i.right_sibling().leaves():
                    units.append(item)
        
        # TODO: assign the neighboring units ---> for ?
        quantityObj['unit']=units
        
        # Related noun phrases
        r_np=[]
        if len(quantity_list)==1:
            def findNP(t):
                if t.height()==2:
                    return
                elif t.label() == 'NP':
                    tmp=t.leaves()
                    r_np.append(' '.join(tmp))
                    return
                else:
                    for child in t:
                        findNP(child)
            findNP(root)
        else:
            r_np_node=find_np(unit_node)
            for np in r_np_node:
                tmp=np.leaves()
                r_np.append(' '.join(tmp))
        quantityObj['noun_phrases'] = r_np
        
        # Rate
        rate=find_rate(unit_node)
        quantityObj['rate'] = rate
        #print(quantityObj)
        
        # Question
        
        # Add to retObj
        retObj[count]=quantityObj
        count+=1
    return retObj


if __name__ == "__main__":
  problem_file = 'number_problems.txt'
  problem_parsed_file = 'number_problems_parsed.txt'
  test_file_path = 'data/cfg_solver_test/'

  with open(test_file_path + problem_file, 'r') as f:
    parsed_sentences = f.read().strip().split('\n')


  os.environ['STANFORD_PARSER'] = 'E://workplace/stanford-parser/jars/stanford-parser.jar'
  os.environ['STANFORD_MODELS'] = 'E://workplace/stanford-parser/jars/stanford-parser-3.9.2-models.jar'

  parser = stanford.StanfordParser(model_path="E:/workplace/stanford-parser/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
  sentences = parser.raw_parse_sents(["John buys 3 candles at a chocolate store", 
  "What is the difference of 22 and 5 ?",
  "What is the result when 6 is divided by the sum of 7 and 5 ?"])
  print(sentences)
  
  for line in sentences:
    for sentence in line:
        p_sentence = ParentedTree.fromstring(str(sentence))
        #print(p_sentence)
        
        retObj = dealWithSentence(p_sentence)
        #p_sentence.draw()