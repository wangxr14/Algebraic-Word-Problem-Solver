import sys
import os
import math
from utils import *
from nltk.parse import stanford
from nltk.tree import ParentedTree
from collections import deque

DEBUG = True
def find_associated_verb(depTree):
    a_verb_nodes = []
    quantities = []
    verb_prefix=['V'] #,'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    quantity_label = ['CD']
    vp_labels=['VP']
    
    # For each node, if the node represents a quantity, uptraverse the tree
    # until a verb is hit
    for nodeId in sorted(depTree.nodes.keys()):
      node = depTree.nodes[nodeId]
      if DEBUG:
        print(node)
      # Find the associated verb for each quantity
      if node['tag'] in quantity_label and isQuantity(node['word']):
        quantities.append(node)
        a_verb_node = depTree.nodes[node['head']] 
        while a_verb_node:
          print('Current ancestor node: ', a_verb_node)
          if a_verb_node['tag'][0] in verb_prefix:
            if DEBUG:
              print(a_verb_node['tag'])
            a_verb_nodes.append(a_verb_node)
            break
          else:
            a_verb_node = depTree.nodes[a_verb_node['head']]
    
    if DEBUG:
      print("Quantities:", quantities)
    return a_verb_nodes, quantities 

#
# Find the subject in a given sentence;
# Assume there is one and only subject and find the first subject in the sentence;
# deal with conjunction later
#        
def find_subject(depTree):
    subject_node = None
    subj_label = ['nsubj']

    for nodeId in sorted(depTree.nodes.keys()):
      node = depTree.nodes[nodeId]
     
      if DEBUG:
        print(node)
    
      if node['rel'] in subj_label:
        subject_node = node 
        break
      
    return subject_node

def find_unit(node):
    units = []
    search_labels=['VP','NP','PP']
    unit_prefix=['N']
    search_queue = deque([node])
    while search_queue:
        cur_node = search_queue.popleft()
        nodes = [cur_node]
        l_sib = cur_node.left_sibling()
        r_sib = cur_node.right_sibling()
        if l_sib:
          nodes.append(l_sib)
        elif r_sib:
          nodes.append(r_sib)

        for node in nodes:
          # Check if the current node is a unit, and if so, add it to the list
          print(node.label())
          if node.label()[0] in unit_prefix and not node.label() in search_labels:
            units.append(node)

          # If not a unit, check if the node is a VP-NP-PP attachment, and
          # if so, add its children to the search queue; if not, continue
          elif node.label() in search_labels:
            for child in node.children:
              search_queue.append(node.child)
    return units
    
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

def find_quantities(pcfg_tree):
    quantity_list=[]
    quantity_label = 'CD'
    def traverse(t):
        if t.height() == 2:
            if t.label() == quantity_label and isQuantity(t.leaves()[0]):
              quantity_list.append(t)
            return
        else:
            for child in t:
                traverse(child)
    traverse(pcfg_tree)
    return quantity_list


def dealWithSentence(pcfgTree, depTree):
    retObj={}
    # Get quantities
    retObj['quantities'] = []
    retObj['quantity_schema'] = {}
    quantity_node_list = find_quantities(pcfgTree)
    if DEBUG:
      print(quantity_node_list)
    subject_node = find_subject(depTree)
    a_verb_nodes, _ = find_associated_verb(depTree)    

    for i, q in enumerate(quantity_node_list):
        quantityObj = {}
        # Detect associated verb
        #a_verb_node = find_associated_verb(node)
        quantityObj['verb'] = a_verb_nodes[i]['word']
        
        # Subject
        if subject_node:
          quantityObj['subject'] = subject_node['word']
        else:
          quantityObj['subject'] = ''
        
        # Unit
        unit_nodes = find_unit(q)
        
        # Followed by PP-NP
        # TODO: assign the neighboring units ---> for ?
        quantityObj['unit'] = [unit.leaves()[0] for unit in unit_nodes]
        
        # Related noun phrases
        r_np=[]
        if len(quantity_node_list)==1:
            def findNP(t):
                if t.height() == 2:
                    return
                elif t.label() == 'NP':
                    tmp = t.leaves()
                    r_np.append(' '.join(tmp))
                    return
                else:
                    for child in t:
                      findNP(child)
            findNP(pcfgTree)
        else:
            for unit_node in unit_nodes:
              r_np_node=find_np(unit_node)
              for np in r_np_node:
                tmp=np.leaves()
                r_np.append(tmp)
        quantityObj['noun_phrases'] = r_np
        # Rate
        rate = [] #find_rate(unit_nodes)
        quantityObj['rate'] = rate
        #print(quantityObj)
        
        # Question
        # Add to retObj
        retObj['quantity_schema'][i] = quantityObj
        retObj['quantities'].append(q.leaves()[0]) 
    return retObj

if __name__ == "__main__":
  problem_file = 'number_problems.txt'
  problem_parsed_file = 'number_problems_parsed.txt'
  test_file_path = 'data/cfg_solver_test/'

  with open(test_file_path + problem_file, 'r') as f:
    parsed_sentences = f.read().strip().split('\n')

  os.environ['STANFORD_PARSER'] = '/Users/liming/nltk_data/stanford-parser-full-2018-10-17/'
  os.environ['STANFORD_MODELS'] = '/Users/liming/nltk_data/stanford-parser-full-2018-10-17/'

  sent = "One had 23 pictures and the other had 32"#"John buys 3 candles and 5 cakes at a chocolate store" 
  pcfgParser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
  depParser = stanford.StanfordDependencyParser()
  
  sentences_pcfg = pcfgParser.raw_parse(sent) 
  pcfg_tree = None
  for line in sentences_pcfg:
    print(line)
    for sentence in line:
        pcfg_tree = ParentedTree.fromstring(str(sentence))
        print(pcfg_tree)
  
  sentences_dep = depParser.raw_parse(sent) 
  depGraph = None
  for res in sentences_dep:
    depGraph = res
  
  print(depGraph)
  '''a_verbs, _ = find_associated_verb(depGraph)
  subj = find_subject(depGraph)
  quantities = find_quantities(pcfg_tree)

  for q in quantities:
    unit_node = find_unit(q)
    print(unit_node)

    #"What is the difference of 22 and 5 ?",
    #"What is the result when 6 is divided by the sum of 7 and 5 ?"])
    #print(sentences)
    
       
  for q, a_verb in zip(quantities, a_verbs):
    print(q.leaves()[0], a_verb['word'], subj['word'])
  '''

  retObj = dealWithSentence(pcfg_tree, depGraph) 
  print(retObj.items())
  #p_sentence.draw()

  
