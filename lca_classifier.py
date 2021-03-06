from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import json
import numpy as np

# TODO: Implement a relevance filter
# TODO: Implement batch training and cross validation
class LCA_Classifier:
  def __init__(self, featFile, relevScore=None, debug=False):
    self.debug = debug
    self.clf = svm.SVC(gamma='auto', probability=True)
    self.mathOps = []
    self.opToId = {}
    self.prblmToQs = {}
    self.relevScore = relevScore
    with open(featFile, 'r') as f:
      featDict = json.load(f) 
     
    featDictTrain = featDict['train']
    featDictVal = featDict['val']
    self.featTrain, self.labelTrain, self.featIdTrain, badIdsTr = self.loadFeatures(featDictTrain)
    self.featVal, self.labelVal, self.featIdVal, badIdsVal = self.loadFeatures(featDictVal)

    # TODO: Write the cleanup version of the feature file back; may need better way
    # to handle corrupted features
    with open(featFile, 'w') as f:
      for i in badIdsTr:
        del featDict['train']['lca_features'][i]
        del featDict['train']['lca_labels'][i]

      for i in badIdsVal:
        del featDict['val']['lca_features'][i]
        del featDict['val']['lca_labels'][i]

      json.dump(featDict, f, indent=4, sort_keys=True)
    

  def loadFeatures(self, featDict): 
    lcafeatDict = featDict['lca_features']
    labelDict = featDict['lca_labels']  
    self.mathOps = featDict['math_ops']
    self.opToId = {op:i for i, op in enumerate(self.mathOps)}
    self.prblmToQs = featDict['quantities']  
     
    feats = []
    labels = []
    featIds = []
    badIds = []
    for pid in range(len(lcafeatDict)):
      
      feats.append([])
      labels.append([])
      # Turns out this is necessary or will lose track of feature order
      featIds.append([])

      n_q = len(lcafeatDict[pid].keys()) - 1
      n_q_relev = len(labelDict[int(pid)].keys())
      
      relevKeys = labelDict[int(pid)].keys() 
      if n_q != n_q_relev:
        # TODO: Implement relevance filter
        #if self.relevScore:
        if self.debug:
          print("Found irrelevant quantities", n_q, n_q_relev, lcafeatDict[pid]['iIndex'], pid)
        # TODO: Find a better way to handle the exception when quantities are not extracted 
        # properly
        if not n_q:
          if self.debug:
            print("Found improper quantity extraction")
          relevKeys = [] 
      # Go through each type of the features for each quantity pairs
      # and convert them into a single vector
      for featId in relevKeys: 
        if featId == 'iIndex': 
          continue
        featIds[pid].append(featId)
        feat = []
        for feat_elem in list(lcafeatDict[pid][featId].values()):
          if isinstance(feat_elem, list):
            feat += feat_elem
            if self.debug:
              print('Find list')
          else:
            feat.append(feat_elem)
        feats[pid].append(feat)
        if self.debug:
          print(labelDict[int(pid)], featId) 

        op = labelDict[int(pid)][featId]
        labels[pid].append(self.opToId[op])
      if self.debug:
        if len(feats[pid]) == 0:
          print(pid, feats[pid], isinstance(feats[pid], list))
      if len(feats) and len(labels):
        feats[pid] = np.array(feats[pid])
        labels[pid] = np.array(labels[pid])
  
    feats_clean = []
    labels_clean = []
    for feat, lbl in zip(feats, labels):
      #if self.debug:
      #  print(feat)

      # TODO: Need better way to handle bad quantity extraction
      if len(list(feat)) == 0:
        if self.debug:
          print("Found bad training example")
        badIds.append(pid)
        continue

      feats_clean.append(feat)
      labels_clean.append(lbl)

    if self.debug:
      print(len(feats_clean), len(labels_clean), feats[0].shape)
    
    return feats_clean, labels_clean, featIds, badIds 
 
  def fit(self, crossValidate=True):
    self.featTrain = np.concatenate(self.featTrain)
    self.labelTrain = np.concatenate(self.labelTrain)

    self.clf.fit(self.featTrain, self.labelTrain)

  def predict(self, scoreFile="data/lca_solver_test/test_lca_scores.json"):
    scoreDict = []
    
    for pid in range(len(self.featVal)):
      labelsPred = self.clf.predict(self.featVal[pid])

      # The SVM automatically shrinks the op ids to the number that actually
      # appears in the dataset, causing discrepancy if some operation does not appear
      # during training; self.idToProbId maps the original op ids to the shrank
      # list of ids in classes_
      self.opIdToProbId = {opId:probId for probId, opId in enumerate(self.clf.classes_)}
      
      scores = self.clf.predict_log_proba(self.featVal[pid])
      #if self.debug:
      #  print(labelsPred, self.clf.classes_)
      #  print(self.mathOps)
      #  print(scores)
      
      scoreDict.append({})
      for j, featId in enumerate(self.featIdVal[pid]):
        for k, op in enumerate(self.mathOps): 
          if k in self.opIdToProbId.keys():
            probId = self.opIdToProbId[k] 
            scoreDict[pid]['_'.join([featId, op])] = scores[j][probId]
    
    scoresInfo = {}
    scoresInfo["quantities"] = self.prblmToQs
    scoresInfo["scores"] = scoreDict
    scoresInfo["math_ops"] = self.mathOps
          
    with open(scoreFile, 'w') as f:
      json.dump(scoresInfo, f, indent=4, sort_keys=True)

  def predictFromFile(self, featFile,
              scoreFile="data/lca_solver_test/test_lca_scores.json"):
    feats, labels, featIds = self.loadFeatures(featFile)
    scoreDict = {}

    for i, pid in enumerate(feats.keys()):
      labelsPred = self.clf.predict(feats[pid])

      # The SVM automatically shrinks the op ids to the number that actually
      # appears in the dataset, causing discrepancy if some operation does not appear
      # during training; self.idToProbId maps the original op ids to the shrank
      # list of ids in classes_
      self.opIdToProbId = {opId:probId for probId, opId in enumerate(self.clf.classes_)}
      
      scores = self.clf.predict_log_proba(feats[pid])
      if self.debug:
        print(labelsPred, self.clf.classes_)
        print(self.mathOps)
        print(scores)
      
      if pid not in scoreDict.keys():
        scoreDict[pid] = {}
      for j, featId in enumerate(featIds[pid]):
        for k, op in enumerate(self.mathOps): 
          if k in self.opIdToProbId.keys():
            probId = self.opIdToProbId[k] 
            scoreDict[pid]['_'.join([featId, op])] = scores[j][probId]
    
    scoresInfo = {}
    scoresInfo["quantities"] = self.prblmToQs
    scoresInfo["scores"] = scoreDict
    scoresInfo["math_ops"] = self.mathOps
          
    with open(scoreFile, 'w') as f:
      json.dump(scoresInfo, f, indent=4, sort_keys=True)

    return labelsPred

  def test(self, feats, labels):
    print(self.clf.score(feats, labels))

if __name__ == "__main__":
  #labelFile = "data/lca_solver_test/label.json" 
  # Create a sample label
  #with open(labelFile, 'w') as f:
  #  json.dump(labels, f)  
  
  featFile = "data/add_sub/features_3fold_0.json"
  
  lcaClf = LCA_Classifier(featFile, debug=True) 
  lcaClf.fit()
  _ = lcaClf.predict()
  lcaClf.test(lcaClf.featVal[0], lcaClf.labelVal[0])
