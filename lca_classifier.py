from sklearn import svm
import json
import numpy as np

# TODO: Implement a relevance filter
class LCA_Classifier:
  def __init__(self, featFileTrain, labelFileTrain, mathOps, debug=False):
    self.debug = debug
    self.clf = svm.SVC(gamma='auto', probability=True)
    self.mathOps = mathOps
    
    # Quantity pairs
    self.pairs = []
    
    # Note: self.feats and self.labels will be converted into numpy array
    # by the loadFeatures function
    self.feats = [] 
    self.labels = []
    self.loadFeatures(featFileTrain, labelFileTrain)

  def loadFeatures(self, featFile, labelFile):
    with open(featFile, 'r') as f:
      featDictAll = json.load(f)
    
    with open(labelFile, 'r') as f:
      labelDictAll = json.load(f)

    for problem in featDictAll.keys():
      featDict = featDictAll[problem]
      labelDict = labelDictAll[problem]
      for pair in featDict.keys():
        q1, q2 = pair[0], pair[1]
        if self.debug:
          print(q1, q2)
        self.pairs.append([q1, q2])    
        self.feats.append(list(featDict[pair].values()))
        self.labels.append(labelDict[pair])
    self.feats = np.array(self.feats)
    self.labels = np.array(self.labels)
    if self.debug:
      print(self.feats, self.labels)
    
  def fit(self):
    self.clf.fit(self.feats, self.labels)

  def predict(self, 
              scoreFile="data/lca_solver_test/lca_score_pred.txt"):
    self.labelsPred = self.clf.predict(self.feats)
    self.probs = self.clf.predict_proba(self.feats)
    with open(scoreFile, 'w') as f:
      for prob, pair in zip(self.probs.tolist(), self.pairs):
        if self.debug:
          print(prob)
        for i, op in enumerate(self.mathOps):
          f.write(' '.join([pair[0], pair[1], op, str(prob[i])]))
          f.write('\n')
    return self.labelsPred

  def test(self, feats, labels):
    print(self.clf.score(feats, labels))

if __name__ == "__main__":
  labelFile = "data/lca_solver_test/label.json" 
  # Create a sample label
  labels = {'0':{'3 5': 1, '3 4': 2, '5 4': 2, '5 3': 1, '4 3': 3, '4 5': 3}}
  with open(labelFile, 'w') as f:
    json.dump(labels, f)  
  
  featFile = "data/lca_solver_test/lca_features.json"
  mathOps = ['+', '-', '-_rev'] #'*', '/', '/_rev']

  lcaClf = LCA_Classifier(featFile, labelFile, mathOps, debug=True) 
  lcaClf.fit()
  _ = lcaClf.predict()
  lcaClf.test(lcaClf.feats, lcaClf.labels)
