import sys
import os
import math
from utils import *
from nltk.parse import stanford
from nltk.tree import ParentedTree
from nltk import pos_tag
from copy import deepcopy

DEBUG = False
TOP = 'TOP'

def find_associated_verb(depTree, pcfgTree):
    a_verb_nodes = []
    quantities = []
    verb_prefix=['V'] #,'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    quantity_label = ['CD']
    tags = pos_tag(pcfgTree.leaves())
    # For each node, if the node represents a quantity, uptraverse the tree
    # until a verb is hit
    for nodeId in sorted(depTree.nodes.keys()):
      node = depTree.nodes[nodeId]
      if node['tag'] == TOP:
        #print('Node without word: ', node)
        continue

      # Find the associated verb for each quantity
      if isQuantity(node['word']):
        quantities.append(node)
        a_verb_node = depTree.nodes[node['head']] 
        found = 0
        while a_verb_node['tag'] != TOP:
          if DEBUG:
            print('Current ancestor node: ', a_verb_node['address'], a_verb_node['word'], a_verb_node['tag'], a_verb_node['rel'])
          #tag = tags[a_verb_node.id]
          if a_verb_node['tag'][0] in verb_prefix:
            if DEBUG:
              print(a_verb_node['tag'])
            
            a_verb_nodes.append(a_verb_node)
            found = 1
            break
          else:
            # Special case: VBZ like "is" may not be the head verb of a
            # quantity node, but rather a sibling of it
            deps = a_verb_node['deps']
            for dep, indices in deps.items():
              for i in indices:
                if DEBUG:
                  print("Current node:", i, depTree.nodes[i]['tag'], depTree.nodes[i]['word'])
                if depTree.nodes[i]['tag'][0] in verb_prefix:
                  a_verb_node = depTree.nodes[i]
                  found = 1
              if found:
                break
            
            if not found:
                a_verb_node = depTree.nodes[a_verb_node['head']]
        if not found:
          a_verb_nodes.append(depTree.nodes[0]) 
    return a_verb_nodes, quantities 

def find_subject(node, depTree):
    subject_node = None
    subj_label = ['nsubj']

    while node['tag'] != TOP:
      deps = node['deps']
      
      for dep, indices in deps.items():
        for i in indices:
          if dep in subj_label:      
            subject_node = depTree.nodes[i]
            return subject_node
      node = depTree.nodes[node['head']]

    print('Subject not found')
    # Return the dummpy TOP node when no subject is found
    return depTree.nodes[0]

def find_unit(node):
    if DEBUG:
      print('Line 81', node)
    unit_node = node
    phrase_labels=['NP']
    unit_patterns = ['of', 'each', 'per']
    while unit_node:
        # If node label is NP, check unit patterns in its siblings:
        # if unit pattern exists, take the parent node as the unit node;
        # otherwise take the current node
        if unit_node.label() in phrase_labels:
          sibling = None
          
          if unit_node.left_sibling():
            sibling = unit_node.left_sibling()
          elif unit_node.right_sibling():
            sibling = unit_node.right_sibling()
          if DEBUG:
            print("Unit node siblings: ", unit_node.parent(), sibling)

          if sibling:
            leaf_set = set(sibling.leaves())
            if leaf_set.intersection(set(unit_patterns)):
              unit_node = unit_node.parent()    
          return unit_node
        else:
          unit_node = unit_node.parent()
    return ParentedTree('', [])
    
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

def find_rate(unit_node, subject_node, depTree):
    rate_words = ['per', 'each']
    rate = []
    if not unit_node:
      leaves = []
    else:
      leaves = unit_node.leaves()    
    
    if DEBUG:
      print(leaves, unit_node)
    # Search the leaves of the unit nodes to find rate patterns
    for i, leaf in enumerate(leaves):
        if leaf in rate_words:
            # If "per" is in the unit, rate is after it 
            if leaf == 'per':
              for leaf2 in leaves[i+1:]:
                if not isQuantity(leaf2):
                  rate.append(leaf2)
              break
            # If "each" appears in the unit, two cases: 
            # 1. "each" at the end, append the nouns before it;
            # 2. "each" before the end, append the nouns after it
            elif i == len(leaves) - 1:
              for leaf2 in leaves[:i]:
                if not isQuantity(leaf2):
                  rate.append(leaf2)
            else:
              for leaf2 in leaves[i+1:]:
                rate.append(leaf2)
    
    # In some cases, rate component is the subject, like "each boy has 5 chocalate bars"
    deps = subject_node['deps']
    found = 0
    for dep, indices in deps.items():
      for i in indices:
        if depTree.nodes[i]['word'] in rate_words:
          rate = [subject_node['word']] 
          found = 1
      if found:
        break
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
    retObj = {}
    # Get quantities
    retObj['quantities'] = []
    retObj['quantity_schema'] = []
    quantity_node_list = find_quantities(pcfgTree)
    if DEBUG:
      print(quantity_node_list)
    a_verb_nodes, _ = find_associated_verb(depTree, pcfgTree)    
   
    for i, q in enumerate(quantity_node_list):
        quantityObj = {}
        a_verb_node = a_verb_nodes[i]
        if a_verb_node['tag'] == TOP:
          print("Verb not found")
          quantityObj['verb'] = ''
        else:
          # Append the position in the sentence to ensure the unique id for each verb mention  
          quantityObj['verb'] = str(a_verb_node['address']) + '_' +  a_verb_node['word']
        
        # Subject
        subject_node = find_subject(a_verb_node, depTree)
        if subject_node != TOP:
            quantityObj['subject'] = subject_node['word']
        else:
            quantityObj['subject'] = ''
        
        # Unit
        unit_node = find_unit(q)
        if DEBUG:
          print(q, unit_node)

        units = []
        
        for item in unit_node.leaves():
          if isQuantity(item):
            continue
          else:
            units.append(item)
        
        quantityObj['unit'] = units  

        # Related noun phrases
        r_np=[]
        if len(quantity_node_list)==1:
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
            findNP(pcfgTree)
        else:
            r_np_node=find_np(unit_node)
            for np in r_np_node:
                tmp=np.leaves()
                r_np.append(' '.join(tmp))
        quantityObj['noun_phrases'] = r_np
        
        # Rate
        rate = find_rate(unit_node, subject_node, depTree)
        quantityObj['rate'] = rate
        #print(quantityObj)
        
        # If unit is not found, use the unit of the quantity before
        if DEBUG:
          print("Unit nodes: ", units, unit_node)
        if not units:
          if DEBUG:
            print("Compensate for empty unit")
          if i > 0:
            quantityObj['unit'] = retObj['quantity_schema'][i-1]['unit']

        # Question      
        # Add to retObj
        retObj['quantity_schema'].append(quantityObj)
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

  pcfgParser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
  depParser = stanford.StanfordDependencyParser()
 
  sent = "Debby had 32 pieces of candy while her sister had 42" #"Each chocalate costs 5 dollars per bar and each cake costs 3 dollars per gram"#"If each question was worth 5 points, what was his final score?" 
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
  
  print(depGraph.tree())
  
  retObj = dealWithSentence(pcfg_tree, depGraph) 
  print(retObj.items())
  #p_sentence.draw()
