from sklearn import svm
import json
import numpy as np

# TODO: Implement a relevance filter
class LCA_Classifier:
  def __init__(self, featFileTrain, mathOps, debug=False):
    self.debug = debug
    self.clf = svm.SVC(gamma='auto', probability=True)
    self.mathOps = mathOps
    self.feats, self.labels, self.featIds, self.prblmToQs = self.loadFeatures(featFileTrain)

  # 
  # Description: Load features stored in a json file with the following format:
  #   Problem_id:
  #
  #     'quantities':
  #       List of all the quantities in the same order it 
  #       appears in the problem in a list
  #     
  #     'features':   
  #       Nq x Nq List of features  
  #     'labels': 
  #       Nq x Nq List of LCA labels
  # 
  # Return: FeatArrs: map problem id to feature array, LabelArrs: map problem id to labels  
  #
  def loadFeatures(self, featFile):
    with open(featFile, 'r') as f:
      featDict = json.load(f) 
    
    lcafeatDict = featDict['lca_features']
    labelDict = featDict['lca_labels']  
    
    feats = {}
    labels = {}
    featIds = {}
    for pid in lcafeatDict.keys():
      if pid not in feats.keys():
        feats[pid] = []
        labels[pid] = []

        # Turns out this is necessary or will lose track of feature order
        featIds[pid] = []
      for featId in lcafeatDict[pid].keys(): 
        featIds[pid].append(featId)
        feats[pid].append(list(lcafeatDict[pid][featId].values()))
        labels[pid].append(labelDict[pid][featId])
      
      feats[pid] = np.array(feats[pid])
      labels[pid] = np.array(labels[pid])
    
    prblmToQs = featDict['quantities']  
    
    #with open(labelFile, 'r') as f:
    #  labelDictAll = json.load(f)
    return feats, labels, featIds, prblmToQs       
 
  def fit(self):
    featAll = np.concatenate([self.feats[pid] for pid in self.feats.keys()])
    labelAll = np.concatenate([self.labels[pid] for pid in self.feats.keys()])
    if self.debug:
      print(featAll.shape, labelAll.shape)
    self.clf.fit(featAll, labelAll)

  def predict(self, featFile,
              scoreFile="data/lca_solver_test/test_lca_scores.json"):
    feats, labels, featIds, prblmToQs = self.loadFeatures(featFile)
    scoreDict = {}

    for i, pid in enumerate(feats.keys()):
      labelsPred = self.clf.predict(feats[pid])
      scores = self.clf.predict_log_proba(feats[pid])
    
      if pid not in scoreDict.keys():
        scoreDict[pid] = {}
      for j, featId in enumerate(featIds[pid]):
        for k, op in enumerate(self.mathOps): 
          scoreDict[pid]['_'.join([featId, op])] = scores[j][k]
    
    scores_with_q = {}
    scores_with_q["quantities"] = prblmToQs
    scores_with_q["scores"] = scoreDict
          
    with open(scoreFile, 'w') as f:
      json.dump(scores_with_q, f)

    return labelsPred

  def test(self, feats, labels):
    print(self.clf.score(feats, labels))

if __name__ == "__main__":
  #labelFile = "data/lca_solver_test/label.json" 
  # Create a sample label
  #with open(labelFile, 'w') as f:
  #  json.dump(labels, f)  
  
  featFile = "data/lca_solver_test/test_features.json"
  mathOps = ['+', '-', '-_rev'] #'*', '/', '/_rev']

  lcaClf = LCA_Classifier(featFile, mathOps, debug=True) 
  lcaClf.fit()
  _ = lcaClf.predict(featFile)
  lcaClf.test(lcaClf.feats['0'], lcaClf.labels['0'])
