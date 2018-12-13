from feature_generator import *
from label_generator import *
from lca_classifier import *
from lca_solver import *
from data_loader import *
from evaluator import *
from utils import *

stage = 1
debug = False
dataPath = "data/multi_arith/"#"data/lca_solver_test/"
schemaFile = "problems_all.json" #"test_problem.json"
problemFile = "../AddSub.json"
equationFile = "../AddSub.json" #"test_equation.json"
mathGrammarFile = "mathGrammar.pcfg"  
featFile = "features.json" #"test_features.json" 
crossValPrefix = "features_6fold" #"test_features_6fold"
lcaScorePrefix = "lca_scores" #"test_lca_scores.json"
predPrefix = "pred"
goldPrefix = "gold"
nFold = 6
feat_choices = {
                'bag-of-words': False,
                'unigram': False,
                'exact_mention': False
                }
constraints = {
              'integer': True,
              'positive': True
              }
if stage < 1:
  # Preprocessing
  print("Preprocessing ...")
  pass

if stage < 2:
  # Feature extraction
  print("Extracting the features ...")
  featureGenerator = FeatureGenerator(dataPath + schemaFile,
                                      dataPath + problemFile, 
                                      dataPath + equationFile, 
                                      dataPath + mathGrammarFile,
                                      debug=debug)
  featureGenerator.extractFeatures(feat_choices)
  featureGenerator.save(dataPath + featFile)

if stage < 3:
  print("Cross-validation split ...")
  featureLoader = LcaFeatureLoader(dataPath + featFile, debug=debug)
  featureLoader.nFoldSplit(nFold, dataPath + crossValPrefix)

if stage < 4:
  # TODO: print more eligible evaluation outputs
  # TODO: save pretrained weights?
  print("Train the LCA classifier ...")
  for i in range(nFold):
    featFileTest = dataPath + crossValPrefix + '_' + str(i) + '.json'
    # LCA classification
    lcaClassifier = LCA_Classifier(featFileTest, debug=debug)
    lcaClassifier.fit()
    print("Crossvalidation test and compute the LCA score ...")
    _ = lcaClassifier.predict(dataPath + lcaScorePrefix + '_' + str(i) + '.json')
    lcaClassifier.test(np.concatenate(lcaClassifier.featVal), np.concatenate(lcaClassifier.labelVal))  

if stage < 5:
  # Find expression tree from LCA
  print("Solving ...")
  for i in range(nFold):
    lcaSolver = LCASolver(dataPath + lcaScorePrefix + '_' + str(i) + '.json', debug=debug, constraints=constraints)
    #lcaSolver = LCASolver(dataPath + lcaScorePrefix + '.json', debug=False)
    trees, lcas = lcaSolver.solve()
    with open(dataPath + predPrefix + '_trees_' + str(i) + '.json', 'w') as f:
      json.dump(trees, f, indent=4, sort_keys=True)
    
    with open(dataPath + predPrefix + '_lcas_' + str(i) + '.json', 'w') as f:
      json.dump(lcas, f, indent=4, sort_keys=True)

if stage < 6:
  # TODO: save the labels in a separate file while generating the features
  for i in range(nFold):
    with open(dataPath + crossValPrefix + '_' + str(i) + '.json', 'r') as f:
      feats = json.load(f)
    labels = feats['val']['lca_labels']
    with open(dataPath + goldPrefix + '_lcas_' + str(i) + '.json', 'w') as f:
      json.dump(labels, f, indent=4, sort_keys=True)
  
  # Evaluate the expressions found by the solver
  prec_avg = 0.
  lcaprec_avg = 0.
 
  for i in range(nFold):
    evaluator = Evaluator(dataPath + goldPrefix + '_lcas_' + str(i) + '.json', 
                          dataPath + predPrefix + '_lcas_' + str(i) + '.json')
    print('###', i, 'Fold Validation Results: ')
    prec = evaluator.precision()
    lcaprec = evaluator.lcaPrecision()
    #recall = evaluator.lcaRecall()
    prec_avg += prec
    lcaprec_avg += lcaprec
    print("Expression Precision: ", prec)
    print("LCA Label Precision: ", lcaprec)
    #print("LCA Label Recall: ", )
  print("Validation Expression Precision: ", prec_avg / nFold)
  print("LCA Label Precision: ", lcaprec_avg / nFold)
