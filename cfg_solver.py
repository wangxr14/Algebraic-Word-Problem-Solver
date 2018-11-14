import sys
import os
import math
from collections import defaultdict

TOP = "TOP"
'''
A grammatical Rule has a probability and a parent category, and is
extended by UnaryRule and BinaryRule
'''
def isQuantity(w):  
  for c in w:
    if c > '9' or c < '0':
      # Check if it is a floating point value or fraction
      if c != '.':
        return 0
  return 1
      
def isMathOp(w, mathOps):
  if w in mathOps:
    return 1
  return 0  

class Rule:
    def __init__(self, probability, parent):
        self.prob = probability
        self.parent = parent

    # Factory method for making unary or binary rules (returns None otherwise)
    @staticmethod
    def createRule(probability, parent, childList):
        if len(childList) == 1:
            return UnaryRule(probability, parent, childList[0])
        elif len(childList) == 2:
            return BinaryRule(probability, parent, childList[0], childList[1])
        elif len(childList) == 3:
            return TenaryRule(probability, parent, childList[0], childList[1], childList[2])

        return None

    # Returns a tuple containing the rule's children
    def children(self):
        return ()

'''
A UnaryRule has a probability, a parent category, and a child category/word
'''


class UnaryRule(Rule):

    def __init__(self, probability, parent, child):
        Rule.__init__(self, probability, parent)
        self.child = child

    # Returns a singleton (tuple) containing the rule's child
    def children(self):
        return (self.child,)  # note the comma; (self.child) is not a tuple


class BinaryRule(Rule):

    def __init__(self, probability, parent, leftChild, rightChild):
        Rule.__init__(self, probability, parent)
        self.leftChild = leftChild
        self.rightChild = rightChild

    # Returns a pair (tuple) containing the rule's children
    def children(self):
        return (self.leftChild, self.rightChild)

class TenaryRule(Rule):

    def __init__(self, probability, parent, leftChild, midChild, rightChild):
        Rule.__init__(self, probability, parent)
        self.leftChild = leftChild
        self.midChild = midChild
        self.rightChild = rightChild

    # Returns a pair (tuple) containing the rule's children
    def children(self):
        return (self.leftChild, self.midChild, self.rightChild)



class Item:

    def __init__(self, label, prob, numparses):
        self.label = label
        self.prob = prob
        self.numparses = numparses

    # returns the node's label
    def toString(self):
        return self.label

class LeafItem(Item):

    def __init__(self, category, value):
        # using log probabilities, this is the default value (0.0 = log(1.0))
        Item.__init__(self, category, 0.0, 1)
        self.val = value
        self.numParses = 1
    
    def toString(self):
        return self.val

'''
An InternalNode stores an internal node in a parse tree (ie, it also
stores pointers to the node's child[ren])
'''

class InternalItem(Item):
    def __init__(self, category, value, prob, children=()):
        Item.__init__(self, category, prob, 0)
        self.children = children
        self.value = value
        
        # Your task is to update the number of parses for this InternalItem
        # to reflect how many possible parses are rooted at this label
        # for the string spanned by this item in a chart
        if len(children) == 1:
          if self.label == TOP:
            self.numParses = children[0].numParses
          else:
            self.numParses = 1
        elif len(children) == 3:
          self.numParses = children[0].numParses * children[1].numParses
        elif len(children) == 0:
          self.numParses = -1  # dummy numParses value; this should not be -1!
        
        else:
          print("Warning: adding a node that has neither 1 or 3 children (CKY may not work correctly)")


    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"

class Cell:
    def __init__(self):
        self.items = {}
    
    def addItem(self, item):
        # Add an Item to this cell
        #pass
        if item.label not in self.items.keys():
          self.items[item.label] = item 
        else:
          # Only keep the maximum backpointer given a label;
          # for ties keep a list of them 
          cur_item = self.items[item.label]
          if cur_item.prob < item.prob:
            # Add the num_parses of the previous max item to the current item 
            item.numParses += self.items[item.label].numParses
            self.items[item.label] = item
          else:
            # Add the num_parses of the current item to the max item
            self.items[item.label].numParses += item.numParses 
      
    def getItem(self, label):
        # Return the cell Item with the given label
        #pass
        if not label in self.items.keys():
          return
        return self.items[label]

    def getItems(self):
        # Return the items in this cell
        return list(self.items.values())

class Chart:
    def __init__(self, sentence):
        # Initialize the chart, given a sentence
        #pass
        self.n = len(sentence)
        self.cells = [[Cell() for i in range(self.n - j)] for j in range(self.n)]
        for i in range(self.n):
          if isQuantity(sentence[i]):
            leafnode = LeafItem('Q', sentence[i])
          else:
            leafnode = LeafItem(sentence[i], sentence[i])
          self.cells[i][0].addItem(leafnode)

    def getRoot(self):
        # Return the item from the top cell in the chart with
        # the label TOP
        #pass
        return self.cells[0][self.n-1].getItem(TOP)

    def getCell(self, i, j):
        # Return the chart cell at position i, j
        #pass
        return self.cells[i][j]

class CFGSolver:
  # Take in an expression sentence, a probability file on the
  def __init__(self, sentence, grammarFile, debug=False):
    self.debug = debug
    self.ckyRules = {}
    self.mathOps = []
    self.readGrammar(grammarFile)
    self.termList = self.extractMathTerms(sentence)  
    if self.debug:
      print(self.termList)

  def readGrammar(self, grammarFile):
      if os.path.isfile(grammarFile):
          file = open(grammarFile, "r")
          for line in file:
              raw = line.split()
              # reminder, we're using log probabilities
              prob = math.log(float(raw[0]))
              parent = raw[1]
              children = raw[
                    3:]   # Note: here, children is a list; below, rule.children() is a tuple
              for c in children:
                # Append any math terms used in the grammar to list 
                if c[0] != 'E' and len(children)==3:
                  self.mathOps.append(c)

              rule = Rule.createRule(prob, parent, children)
              if rule.children() not in self.ckyRules:
                  self.ckyRules[rule.children()] = set([])
              self.ckyRules[rule.children()].add(rule)

  def CKY(self):
    # Parse with CFG in the form E -> ABC and E -> A
    # TODO: allow more than one best trees
    # TODO: allow arbitrary word combining order 
    self.chart = Chart(self.termList)
    n = self.chart.n
    
    # Only even-indexed diagonal will be used
    for i in range(0, n, 2):
      for j in range(n-i): 
        for k in range(i-1):
          for l in range(i-k-1):
            if self.debug:
              print(j, j+k+1, j+k+l+2)
            lCell = self.chart.getCell(j, k).getItems()
            mCell = self.chart.getCell(j+k+1, l).getItems()
            rCell = self.chart.getCell(j+k+l+2, i-k-l-2).getItems()
            for lChild in lCell:
              for mChild in mCell:
                for rChild in rCell:
                  if self.debug:
                      print(lChild.label, mChild.label, rChild.label)
                    
                  if (lChild.label, mChild.label, rChild.label) in self.ckyRules.keys():
                    rules = self.ckyRules[(lChild.label, mChild.label, rChild.label)]
                    for rule in rules:
                      rule_prob, parent = rule.prob, rule.parent
                      parent_prob = lChild.prob + mChild.prob + rChild.prob + rule_prob 
                      parentNode = InternalItem(parent, '', parent_prob, children=(lChild, mChild, rChild))    
                      if self.debug:
                        print(parentNode.label)
                      self.chart.cells[j][i].addItem(parentNode)
        
        # Add quantity label to the quantities using uninary rule E -> Q
        if i == 0:
          cell = self.chart.getCell(j, i).getItems()
          for child in cell:
            # Check if a child's label satifies any unitary rule 
            if (child.label,) in self.ckyRules.keys():
              if self.debug:
                print(child.label)
            
              rules = self.ckyRules[(child.label,)]
              for rule in rules:
                prob, parent = rule.prob, rule.parent
                node = InternalItem(parent, '', prob, children=(child,))
                self.chart.cells[j][i].addItem(node)

    topItems = self.chart.getCell(0, n-1).getItems()
    if len(topItems) == 0:
      return
    else:
      sortedItems = sorted(topItems, key=lambda x:x.prob, reverse=True) 
      bestNode = sortedItems[0]
      prob = bestNode.prob
      topNode = InternalItem(TOP, '', prob, children=(bestNode,)) 
      self.chart.cells[0][n-1].addItem(topNode)
    return self.chart.getRoot() 

  def extractMathTerms(self, sentence):
    termList = []
    for w in sentence:
      if self.debug:
        print(w)
      if isQuantity(w):
        termList.append(w)
      elif isMathOp(w, self.mathOps):
        termList.append(w)

    if self.debug:
      print(termList)
    return termList

'''              for op in ops:
                if checkLCA(i, j, k, op):
                  score = scoreLCA(i, j, k, op)
                  op_node = InternalItem(op, score, children=(lChild, rChild))  
                  self.chart.cells[j][i].addItem(op_node)
      #def AstarParse(self):
  def checkLCA(self, i, j, k, op):
    if k >= i:
      raise ValueError("Invalid subtree index") 
    for iLeft in range(k):
      for iRight in range(j-k):

  def findLCA(self):
  
  def scoreLCA(self):

  def findQuantities(self, problem_text):
    words = problem_text.strip().split()
    for w in words:
      is_quantity = 1
      
      for c in w:
        if c > '9' or c < '0':
          # Check if it is a floating point value or fraction
          if c != '.':
            is_quantity = 1 
      
      if is_quantity:
        self.qs.append(w)

  def findExplicitLCA(self, problem_text):
    words = problem_text.strip().split()
    for i, w in enumerate(words):
      # If names of the arithmetic operation are seen in the text, assign
      # the proper category to it with probability 1
      # For addition and multiplication, order does not matter so two 
      # possible pair orders are both assigned the same label; for 
      # subtraction and division only the rightly ordered pair is assigned
      # a label
      if w == "add" or w == "adds":
        self.q2lca[(words[i-1], words[i+1])] = ('+', 1.0)
        self.q2lca[(words[i+1], words[i-1])] = ('+', 1.0)

      elif w == "subtract" or w == "subtracts" or words[i:i+1] == "subtracted by":
        self.q2lca[(words[i-1], words[i+1])] = ('-', 1.0)
      elif w == "multiply" or w == "multiplies":
        self.q2lca[(words[i-1], words[i+1])] = ('*', 1.0)
      elif w == "divide" or w == "divides":
        self.q2lca[(words[i-1], words[i+1])] = '/'

  def LCAToTree(self,):
'''

if __name__ == "__main__":
  #sentence = "5.5 multiplies 3 divides 2 adds 4"
  problem_file = 'number_problems.txt'
  problem_parsed_file = 'number_problems_parsed.txt'
  with open(problem_file, 'r') as f:
    sentences = f.read().strip().split('\n')

  with open(problem_parsed_file, 'w') as f:
    for sentence in sentences:
      sentence = sentence.split()
      parser = CFGSolver(sentence, "mathgrammar.pcfg", debug=True)
      tree = parser.CKY()
      if tree: 
        print(tree.toString())
      else:
        print("Fail to parse")
