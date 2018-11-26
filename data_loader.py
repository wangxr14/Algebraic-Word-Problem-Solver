import json

class LcaFeatureLoader:
  def __init__(self, featFile, debug=False):
    self.debug = debug
    self.featDict = {}
    self.crossValDict = {}
    with open(featFile, 'r') as f:
      self.featDict = json.load(f)

  def nFoldSplit(self, nFold, prefix='features'):
    self.valSize = int(len(self.featDict['lca_features']) / nFold)
    self.nFold = nFold
    featAll = self.featDict['lca_features']
    labelAll = self.featDict['lca_labels']
    quantityAll = self.featDict['quantities'] 
    mathOps = self.featDict['math_ops'] 

    for i in range(self.nFold):
      featVal = featAll[i*self.valSize:(i+1)*self.valSize] 
      labelVal = labelAll[i*self.valSize:(i+1)*self.valSize] 
      quantityVal = quantityAll[i*self.valSize:(i+1)*self.valSize] 

      featTrain = featAll[:i*self.valSize] + featAll[(i+1)*self.valSize:] 
      labelTrain = labelAll[:i*self.valSize] + labelAll[(i+1)*self.valSize:] 
      quantityTrain = quantityAll[(i-1)*self.valSize:i*self.valSize] + quantityAll[(i+1)*self.valSize:] 
      if self.debug:
        print(self.valSize)
        print(len(featTrain), len(labelTrain), len(featAll[(i+1)*self.valSize:]))

      crossValDict = {}
      crossValDict['train'] = {
        'lca_features': featTrain,
        'lca_labels': labelTrain,
        'quantities': quantityTrain,
        'math_ops': mathOps
      }
      crossValDict['val'] = {
        'lca_features': featVal,
        'lca_labels': labelVal,
        'quantities': quantityVal,
        'math_ops': mathOps
      }
      crossValFile = prefix + '_' + str(i) + '.json'

      with open(crossValFile, 'w') as f:
        json.dump(crossValDict, f)

if __name__ == "__main__":
  featFile = "data/lca_solver_test/test_features.json"
  loader = LcaFeatureLoader(featFile, 6, debug=True)
  loader.nFoldSplit('data/lca_solver_test/test_features_6fold') 
