from collections import deque
from heapq import *
from classes import *
import json

#TODO: Impliment functions to extract pair score from grammar file 
#TODO: Impliment functions to compute pair score from features
#TODO: Add additional constraints to the heuristic for the beam search
#TODO: Implement a function to check if a tree is monotonic
class LCASolver:
  
  def __init__(self, lcaScoreFile, reversible=['+', '*'], debug=False, constraints=None):
    self.debug = debug
    self.lcaScores, self.qs, self.mathOps = self.readScore(lcaScoreFile)
    self.qNodes = []
    self.reversible = reversible
    self.constraints = constraints

  def readScore(self, lcaScoreFile):
    qToLcaScores = {}
    with open(lcaScoreFile, 'r') as f:
      '''line = f.read().strip().split('\n')
      for raw in line:
        raw = raw.split()
        q1, q2, op, score = raw[0], raw[1], raw[2], float(raw[3])
        qToLcaScores[(q1, q2, op)] = score 
      '''
      scoreDict = json.load(f)
      qToLcaScores = scoreDict["scores"]
      qs = scoreDict["quantities"]
      mathOps = scoreDict["math_ops"]
    return qToLcaScores, qs, mathOps
  
  def solve(self, beamWidth=100):
    trees = []
    lcas = []
    for pid in range(len(self.lcaScores)):  
      self.qNodes = []
      for i, q in enumerate(self.qs[pid]):
        # Store both the position index and the value of the quantity in 
        # a tuple as category
        self.qNodes.append(Node(str('_'.join([str(i), q])), 0.))
      tree = self.beamSearch(self.lcaScores[pid], beamWidth)
      if self.debug:
        print(tree[1][0].toString())
      trees.append(tree[1][0].toString())
      lcas.append(self.findLcas(tree[1][0]))
    return trees, lcas

  #
  # Perform beam search to find trees with the highest scores;
  # in each iteration, goes through all the possible pairs of
  # expression subtree nodes in the state and merge them to form 
  # a new, bigger subtree until a single tree is formed; the score
  # for a tree is the LCA score of all the quantity pairs (leaf nodes)
  #
  def beamSearch(self, lcaScore, beamWidth=100):
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

      #if self.debug:
      #  print("Current state:")
      #  self.printState(score_state)
      for q1 in s:
        for q2 in s:
          if q1.leafIds != q2.leafIds:
            for op in self.mathOps:
              score_state_next = self.merge(lcaScore, q1, q2, score_state, op)
              # Check if the current state is complete and if so,
              # whether it satisfies the constraints
              if len(score_state_next[1]) == 1:
                if not self.checkConstraints(score_state_next):
                  continue

              #if self.debug:
              #  print("Next state:")
              #  self.printState(score_state_next)
            
              self.state_queue.append(score_state_next)
              # Notice here state_queue has the type list after being sorted
      self.state_queue = sorted(self.state_queue, key=lambda x:x[0], reverse=True)
      if len(self.state_queue) > beamWidth:
        self.state_queue = self.state_queue[:beamWidth]

      # Convert the state queue back to deque
      self.state_queue = deque(self.state_queue)
    #if self.debug:
    #  print(self.printState(self.state_queue[0]))
    return self.state_queue[0]

  def finish(self):
    for s in self.state_queue:
      if len(s[1]) > 1:
        return 0
    return 1

  # TODO: Need more general way of checking constraints
  def checkConstraints(self, s):
    if not self.constraints['integer'] and not self.constraints['positive']:
      return 1
    
    if len(s[1]) == 1:
      res = s[1][0].eval()
      if self.debug:
        print(res, s[1][0].toString())
      if self.constraints['integer']:
        if self.constraints['positive']:
          if res == int(res) and res > 0:
            return 1
            
        elif res == int(res):
          if self.debug:
            print('Constraint not satisfied')
          return 1
      else:
        if self.constraints['positive']:
          if res > 0:
            return 1
    else:
      print('Incomplete state!')
      return 0
    if self.debug:
      print('State do not satisfy the constraints')
    return 0


  def merge(self, lcaScore, q1, q2, score_state, op):
    score = score_state[0]
    s = score_state[1] 
    # Sufficient to compute LCA score increment since the old score within
    # the node has already been added
    score_increment = self.lcaScoreIncrement(lcaScore, q1, q2, op)
    opNode = Node(op, score_increment, children=(q1, q2))
    s_next = [q for q in s if q not in [q1, q2]]
    # Compute the new state after merging the nodes 
    s_next.append(opNode)
    score += score_increment
    return (score, s_next) 
  
  def lcaScoreIncrement(self, lcaScore, q1, q2, op):
    score = 0
    for id1 in q1.leafIds:
      for id2 in q2.leafIds:
        scoreId = '_'.join([id1.split('_')[0], id2.split('_')[0], op])
        if scoreId in lcaScore.keys(): 
          score += lcaScore[scoreId]   
        else:
          score += float('-inf')
    return score

  #def checkMonotonic(self, state): 
  def printState(self, state):
    score = state[0]
    state_nodes = []
    for node in state[1]:
      state_nodes.append([node.leafIds, node.label])
    print(score, state_nodes)

  def findLcas(self, node):
    qToLcas = {}
    node_queue = deque([node])
    while node_queue:
      cur_node = node_queue.popleft()
      if len(cur_node.children) > 2:
        raise ValueError('Number of children is asserted to be at most 2')
      # Check if the current node is a leaf node
      elif not len(cur_node.children):
        continue
      l_child, r_child = cur_node.children
      for q_l in l_child.leafIds:
        for q_r in r_child.leafIds:
          pos_l = q_l.split('_')[0]
          pos_r = q_r.split('_')[0]
          qToLcas[pos_l + '_' + pos_r] = cur_node.label
          if cur_node.label in self.reversible:
            qToLcas[pos_r + '_' + pos_l] = cur_node.label
          else:
            if cur_node.label.split('_')[-1] == 'rev':
              qToLcas[pos_r + '_' + pos_l] = cur_node.label.split('_')[0]
            else:
              qToLcas[pos_r + '_' + pos_l] = cur_node.label + '_rev'
          

      node_queue.append(l_child)
      node_queue.append(r_child)
  
    return qToLcas

if __name__ == '__main__':
  lcaScoreFile = "data/lca_solver_test/test_lca_scores.json"
  # '-rev' and '/rev' denote scores when the order of two quantities is reversed
  # mathOps = ['+', '-', '*', '/', '-_rev', '/_rev']
  lca_solver = LCASolver(lcaScoreFile, debug=True)
  bestTree, qToLcas = lca_solver.solve(beamWidth=200)
  if len(bestTree) == 0:
    print("Fail to parse")
  
  else:
    print(bestTree, qToLcas) 
