from feature_generator import *
from lca_classifier import *
from lca_solver import *
from data_loader import *
from utils import *

stage = 1
dataPath = "data/lca_solver_test/"
problemFile = "test_problem.json"
equationFile = "test_equation.json"
mathGrammarFile = "mathGrammar.pcfg"  
featFile = "test_features.json" 
crossValPrefix = "test_features_6fold"
lcaScoreFile = "test_lca_scores.json"
nFold = 6

if stage < 1:
  # Preprocessing
  pass

if stage < 2:
  # Feature extraction
  featureGenerator = FeatureGenerator(dataPath + problemFile, 
                                      dataPath + equationFile, 
                                      dataPath + mathGrammarFile,
                                      debug=False)
  featureGenerator.extractFeatures()
  featureGenerator.save(dataPath + featFile)

if stage < 3:
  featureLoader = LcaFeatureLoader(dataPath + featFile)
  featureLoader.nFoldSplit(nFold, crossValPrefix)

if stage < 4:
  for i in range(nFold):
    featFileTest = dataPath + crossValPrefix + '_' + str(i) + '.json'
    # LCA classification
    lcaClassifier = LCA_Classifier(featFileTest)
    lcaClassifier.fit()
    _ = lcaClassifier.predict(dataPath + lcaScoreFile)
  

    # Find expression tree from LCA
    lcaSolver = LCASolver(dataPath + lcaScoreFile, debug=False)
    trees = lcaSolver.solve()
    print("Predicted Expression: ", trees[0][1][0].toString())
    #printTree(trees) 
