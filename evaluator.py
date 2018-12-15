from nltk import precision
from nltk import recall
import json
from label_generator import *

class Evaluator:
  def __init__(self, 
              goldFile, 
              predFile, 
              mathOps=['+', '-', '*', '/', '-_rev', '/_rev'], 
              debug=False):
    self.debug = debug
    self.mathOps = mathOps 
    with open(goldFile, 'r') as f:
      self.goldLcas = json.load(f)
    
    with open(predFile, 'r') as f:
      self.predLcas = json.load(f)

  # TODO
  def convertReverseOp(self):
    pass

  def precision(self):
    prec = 0.
    tot = len(self.goldLcas)
    if self.debug:
      print(tot, len(self.predLcas))
    for i in range(tot):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      curPrec = precision(set(gold), set(pred))
      if curPrec == 1.:
        prec += 1

    return prec / tot

  def lcaPrecision(self):
    prec = 0.
    tot = len(self.goldLcas)
    for i in range(len(self.goldLcas)):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      curPrec = precision(set(gold), set(pred))
      if self.debug:
        print(curPrec)
      # TODO: Figure out what is wrong for some
      # LCAs to be None type
      if curPrec:
        prec += curPrec

    return prec / tot
    

  def lcaRecall(self):
    rec = 0.
    tot = len(self.goldLcas)
    for i in range(len(self.goldLcas)):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      rec += recall(set(pred), set(gold))
    return rec / tot

if __name__ == "__main__":
  #labelGen = LcaLabelGenerator("data/lca_solver_test/mathGrammar.pcfg")
  #_, lcaLabels = labelGen.generate("X=((3+5)-4)")
  #with open("data/lca_solver_test/gold.json", 'w') as f:
  #  json.dump([lcaLabels], f)
  nFold = 5
  gold_pre = "data/single_op/pair_feat_pos_constraint/gold_lcas" 
  #"data/multi_arith/experiments/pair_feat_with_pos_heuristic_dec13/gold_lcas"
  pred_pre = "data/single_op/pair_feat_pos_constraint/pred_lcas"
 
  prec = 0.
  lcaprec = 0. 
  for i in range(nFold):
    evaluator = Evaluator(gold_pre + "_{}.json".format(str(i)), pred_pre + "_{}.json".format(str(i)))
    print(evaluator.precision())
    print(evaluator.lcaPrecision())
    prec += evaluator.precision()
    lcaprec += evaluator.lcaPrecision() 
    #print(evaluator.lcaRecall())

  print("Avg precision:", prec / nFold)
  print("Avg lcaprecision:", lcaprec / nFold)
