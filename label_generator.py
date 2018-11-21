from classes import *
from utils import *
from cfg_solver import *

DEBUG = False
# TODO: A function to change the tree into a monotonic one
class LcaLabelGenerator:
  def __init__(self, mathGrammarFile, debug=False):
    self.debug = debug
 
  def generate(self, equation):
    self.cfgSolver = CFGSolver(equation, mathGrammarFile, debug=debug)
    self.tree = self.cfgSolver.CKY() 
    if self.debug:
      print(self.tree.toString())
    if not self.tree:
      print("Fail to parse")
 
    return findLcas(self.tree) 

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

def findLcas(tree, pairToLcas={}, mathOps=['+', '-', '*', '/']):    
  children = tree.children
  if tree.label == "TOP":
    if DEBUG:
      print(children[0].label)
    pairtolcas = findLcas(children[0], pairToLcas)
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
      pairToLcas = findLcas(lChild, pairToLcas)
      pairToLcas = findLcas(rChild, pairToLcas)
    else:
      mChild = children[1]
      pairToLcas = findLcas(mChild, pairToLcas)
    
  return pairToLcas


if __name__ == "__main__":
  equation = "( ( 13.0 + 5.0 ) - 10.0 )"
  equation = equation.split()
  labelGen = LcaLabelGenerator(equation, "data/lca_solver_test/mathGrammar.pcfg", debug=False)
  labels = labelGen.generate()
  #print(findLeafIds(labelGen.tree))
  print(labels.items())
