from nltk import precision
from nltk import recall
import json
from label_generator import *

class Evaluator:
  def __init__(self, goldFile, predFile, mathOps=['+', '-', '*', '/', '-_rev', '/_rev']):
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

    for i in range(tot):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      curPrec = precision(set(gold), set(pred))
      if curPrec == 1.:
        prec += 1

    return prec / tot

  def lcaPrecision(self):
    preds = []
    golds = []
    for i in range(len(self.goldLcas)):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      preds += pred
      golds += gold
    return precision(set(golds), set(preds))
    

  def lcaRecall(self):
    precision = 0.
    preds = []
    golds = []
    for i in range(len(self.goldLcas)):
      pred = self.predLcas[i].items()
      gold = self.goldLcas[i].items()
      preds += pred
      golds += gold
    return recall(set(golds), set(preds))

if __name__ == "__main__":
  labelGen = LcaLabelGenerator("data/lca_solver_test/mathGrammar.pcfg")
  _, lcaLabels = labelGen.generate("X=((3+5)-4)")
  with open("data/lca_solver_test/gold.json", 'w') as f:
    json.dump([lcaLabels], f)

  evaluator = Evaluator("data/lca_solver_test/gold.json", "data/lca_solver_test/gold.json")
  print(evaluator.precision())
  print(evaluator.lcaPrecision())
  print(evaluator.lcaRecall())
