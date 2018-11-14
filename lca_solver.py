from collections import deque
from heapq import *
import json

class Item:

    def __init__(self, label, prob, numParses):
        self.label = label
        self.prob = prob
        self.numParses = numParses

    # Returns the node's label
    def toString(self):
        return self.label


class Node(Item):

    def __init__(self, category, score, children=()):
        Item.__init__(self, category, score, 0)
        self.children = children
        self.score = score
        self.leafIds = []
        # Keep a list of indices for the quantities in the subtree
        if len(children) == 0:
          self.leafIds = [category]
        else:
          for c in children:
            self.leafIds += c.leafIds 
        

        if len(children) == 1:
          if category == TOP:
            self.numParses = children[0].numParses
          else:
            self.numParses = 1
        elif len(children) == 2:
          self.numParses = children[0].numParses * children[1].numParses
        elif len(children) == 0:
          self.numParses = -1  # dummy numParses value; this should not be -1!
        
        elif len(self.children) > 2:
            print("Warning: adding a node with more than two children (CKY may not work correctly)")

    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"

#TODO: Impliment functions to extract pair score from grammar file 
#TODO: Impliment functions to compute pair score from features
#TODO: Add additional constraints to the heuristic for the beam search
class LCASolver:
  
  def __init__(self, qs, mathOps, lcaScoreFile, debug=False):
    self.debug = debug
    self.qNodes = []
    for q in qs:
      self.qNodes.append(Node(q, 0.))
    self.mathOps = mathOps
    self.qToLcaScores = self.readScore(lcaScoreFile)
    if debug:
      print(self.qToLcaScores.keys())
    

  def readScore(self, lcaScoreFile):
    qToLcaScores = {}
    with open(lcaScoreFile, 'r') as f:
      line = f.read().strip().split('\n')
      for raw in line:
        raw = raw.split()
        q1, q2, op, score = raw[0], raw[1], raw[2], float(raw[3])
        qToLcaScores[(q1, q2, op)] = score 
    
    return qToLcaScores

  #
  # Perform beam search to find trees with the highest scores;
  # in each iteration, goes through all the possible pairs of
  # expression subtree nodes in the state and merge them to form 
  # a new, bigger subtree until a single tree is formed; the score
  # for a tree is the LCA score of all the quantity pairs (leaf nodes)
  #
  def beamSearch(self, beamWidth=100):
    self.state_queue = deque()
    self.state_queue.append((0., self.qNodes))
    while not self.finish():
      score_state = self.state_queue.popleft()
      score = score_state[0]
      s = score_state[1]
      # If a tree is already formed in the current state, skip it;
      if len(s) == 1:
        self.state_queue.append(score_state)
        continue

      if self.debug:
        print("Current state:")
        self.printState(score_state)
      for q1 in s:
        for q2 in s:
          if q1.leafIds != q2.leafIds:
            for op in self.mathOps:
              score_state_next = self.merge(q1, q2, score_state, op)
              if self.debug:
                print("Next state:")
                self.printState(score_state_next)

              self.state_queue.append(score_state_next)
              # Notice here state_queue has the type list after being sorted
      self.state_queue = sorted(self.state_queue, key=lambda x:x[0], reverse=True)
      if len(self.state_queue) > beamWidth:
        self.state_queue = self.state_queue[:beamWidth]

      # Convert the state queue back to deque
      self.state_queue = deque(self.state_queue)
    if self.debug:
      print(self.state_queue)
    return self.state_queue[0]

  def finish(self):
    for s in self.state_queue:
      if len(s[1]) > 1:
        return 0
    return 1

  def merge(self, q1, q2, score_state, op):
    score = score_state[0]
    s = score_state[1] 
    # Sufficient to compute LCA score increment since the old score within
    # the node has already been added
    score_increment = self.lcaScoreIncrement(q1, q2, op)
    opNode = Node(op, score_increment, children=(q1, q2))
    s_next = [q for q in s if q not in [q1, q2]]
    # Compute the new state after merging the nodes 
    s_next.append(opNode)
    score += score_increment
    return (score, s_next) 
  
  def lcaScoreIncrement(self, q1, q2, op):
    score = 0
    for i in q1.leafIds:
      for j in q2.leafIds:
        if (i, j, op) in self.qToLcaScores.keys():
          if self.debug and (i, j, op) == ['3', '5', '*']:
            print(i, j, op)
          score += self.qToLcaScores[(i, j, op)]   
        else:
          score += float('-inf')
    return score

  def printState(self, state):
    score = state[0]
    state_nodes = []
    for node in state[1]:
      state_nodes.append([node.leafIds, node.label])
    print(score, state_nodes)

if __name__ == '__main__':
  lcaScoreFile = "lca_scores.txt"
  # '-rev' and '/rev' denote scores when the order of two quantities is reversed
  mathOps = ['+', '-', '*', '/', '-_rev', '/_rev']
  qs = ['5', '3', '4']
  lca_solver = LCASolver(qs, mathOps, lcaScoreFile, debug=True)
  bestTree = lca_solver.beamSearch(beamWidth=3)
  if len(bestTree) == 0:
    print("Fail to parse")
  
  else:
    print(bestTree[0], bestTree[1][0].toString())   
