import json
from collections import deque
from classes import *
from utils import *
from cfg_solver import *

DEBUG = False
# TODO: A function to change the tree into a monotonic one
class LcaLabelGenerator:
  def __init__(self, mathGrammarFile, mathOps=['+', '-', '*', '/', '-_rev', '/_rev'], debug=False):
    self.debug = debug
    self.mathOps = mathOps
    self.mathGrammarFile = mathGrammarFile
 
  def generate(self, equation):
    self.cfgSolver = CFGSolver(equation, self.mathGrammarFile, debug=self.debug)
    tree = self.cfgSolver.CKY() 
    if not tree:
      print("Fail to parse")
    elif self.debug:
      print(tree.toString())
 
    return tree, findLcas(tree, self.mathOps) 

  def generateFromFile(self, equationFile):
    labelsDict = {}
    labelsAll = []

    with open(equationFile, 'r') as f:
      equationInfos = json.load(f)
    
    for equationInfo in equationInfos:
      equation = equationInfo["lEquations"][0].split('=')[-1]
      eqTokens = tokenizeEq(equation)
      if self.debug:
        print(eqTokens)
      _, labels = self.generate(eqTokens)
      
      #if self.debug:
      #  print(labels)      
      labelsAll.append(labels)
    labelsDict['labels'] = labelsAll
    labelsDict['mathOps'] = self.mathOps
    return labelsDict

  #def monotonicize(self):

def findLeafIds(tree, leafIds=[]):
  children = tree.children
  if tree.label == "TOP":
    #if DEBUG:
    #  print(children[0].label)  
    leafIds += findLeafIds(children[0])
  elif len(children) == 3:
    if DEBUG:
      print(tree.label)    
    leafIds = findLeafIds(children[0]) + findLeafIds(children[1]) + findLeafIds(children[2])
  elif len(children) == 1:
    child = children[0]
    if DEBUG:
      print(tree.children[0].label)
    if '_' in child.val:
      if DEBUG:
        print(child.label)
      leafIds = [child.val.split('_')[0]] 
  
  return leafIds

def findLcas(tree, mathOps=['+', '-', '*', '/']):    
  queue = deque([tree])
  pairToLcas = {}
  while len(queue): 
    node = queue.popleft()
    children = node.children

    if node.label == "TOP":
      if DEBUG:
        print(children[0].label)
      queue.append(children[0])  
      #pairtolcas = findLcas(children[0], pairToLcas)
    if len(children) == 3:
      # Check the label of the middle child, which may contain the operation
      # type between the two subtrees
      opLabel = children[1].label
      if opLabel in mathOps:
        lChild = children[0]
        rChild = children[2]
        if DEBUG:
          print(lChild.label, children[1].label, rChild.label)
        for ql in findLeafIds(lChild):
          for qr in findLeafIds(rChild):
            if DEBUG:
              print(ql, qr)
            pairToLcas['_'.join([ql, qr])] = opLabel
            # For noncommutative operations, add reverse operation labels
            if opLabel == '-' or opLabel == '/':
              pairToLcas['_'.join([qr, ql])] = opLabel + '_rev'
            else:
              pairToLcas['_'.join([qr, ql])] = opLabel

        queue.append(lChild)
        queue.append(rChild)
        
      else:
        mChild = children[1]
        queue.append(mChild)
    
  return pairToLcas


if __name__ == "__main__":
  equation = "( ( 13.0 + 5.0 ) - 10.0 )"
  equation = equation.split()
  labelGen = LcaLabelGenerator("data/lca_solver_test/mathGrammar.pcfg", debug=False)
  #labels = labelGen.generate(equation)
  labelsDict = labelGen.generateFromFile("data/lca_solver_test/test_equation.json")
  #print(findLeafIds(labelGen.tree))
  #with open("data/test_lca_solver/labels.txt", 'r') as f:
  for labels in labelsDict['labels']:
    print(labels.items())
