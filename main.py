from feature_generator import *
from lca_classifier import *
from lca_solver import *
from data_loader import *
from utils import *

stage = 1
dataPath = "data/multi_arith/"#"data/lca_solver_test/"
problemFile = "problems.json" #"test_problem.json"
equationFile = "../MultiArith.json" #"test_equation.json"
mathGrammarFile = "mathGrammar.pcfg"  
featFile = "features.json" #"test_features.json" 
crossValPrefix = "features_6fold" #"test_features_6fold"
lcaScoreFile = "lca_scores.json" #"test_lca_scores.json"
predPrefix = "pred"
nFold = 6

if stage < 1:
  # Preprocessing
  print("Preprocessing ...")
  pass

if stage < 2:
  # Feature extraction
  print("Extracting the features ...")
  featureGenerator = FeatureGenerator(dataPath + problemFile, 
                                      dataPath + equationFile, 
                                      dataPath + mathGrammarFile,
                                      debug=False)
  featureGenerator.extractFeatures()
  featureGenerator.save(dataPath + featFile)

if stage < 3:
  print("Cross-validation split ...")
  featureLoader = LcaFeatureLoader(dataPath + featFile, debug=True)
  featureLoader.nFoldSplit(nFold, dataPath + crossValPrefix)

if stage < 4:
  print("Train the LCA classifier ...")
  for i in range(nFold):
    featFileTest = dataPath + crossValPrefix + '_' + str(i) + '.json'
    # LCA classification
    lcaClassifier = LCA_Classifier(featFileTest)
    lcaClassifier.fit()
    print("Crossvalidation test and compute the LCA score ...")
    _ = lcaClassifier.predict(dataPath + lcaScoreFile)
  

    # Find expression tree from LCA
    print("Solving ...")
    lcaSolver = LCASolver(dataPath + lcaScoreFile, debug=False)
    trees, lcas = lcaSolver.solve()
    print("Predicted Expression: ", trees)
    with open(dataPath + predPrefix + '_trees.json', 'w') as f:
      json.dump(trees, f)
    
    with open(dataPath + predPrefix + '_lcas.json', 'w') as f:
      json.dump(lcas, f)


    #printTree(trees) 
