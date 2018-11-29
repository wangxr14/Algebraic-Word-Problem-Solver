import json
import os
from nltk.tree import ParentedTree
from nltk.parse import stanford
from quantity_schema import *
from utils import *

class Preprocessor:
  def __init__(self, rawProblemFile, debug=False):
    self.debug = debug
    with open(rawProblemFile, 'r') as f:
      self.problems = json.load(f)
    os.environ['STANFORD_PARSER'] = '/Users/liming/nltk_data/stanford-parser-full-2018-10-17/'
    os.environ['STANFORD_MODELS'] = '/Users/liming/nltk_data/stanford-parser-full-2018-10-17/'

    self.pcfgParser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    self.depParser = stanford.StanfordDependencyParser() 
    
  def prepare(self, problemFile):
    prblmDict = self.extractQuantitySchema()
    with open(problemFile, 'w') as f:
      json.dump(prblmDict, f)
  
  def extractQuantitySchema(self):
    qSchemas = []
    for problem in self.problems:
      sents = problem['sQuestion'].split('.')
      pSents = self.pcfgParser.raw_parse_sents(sents)
      dSents = self.depParser.raw_parse_sents(sents)
      
      schemas = {'quantities':[], 'quantity_schema':[]}
      for pline, dline in zip(pSents, dSents):
        for psent, dsent in zip(pline, dline):
          pcfgTree = ParentedTree.fromstring(str(psent)) 
          depTree = dsent 
          schema = dealWithSentence(pcfgTree, depTree) 
          schemas = self.addSchema(schema, schemas)
          
      schemas['question'] = self.findQuestion(sents)

      if self.debug:
        quantities = schemas['quantities']
        print(quantities, schemas.items())
      qSchemas.append(schemas)
    return qSchemas

  def findQuestion(self, sents):
    # Find the last sentence
    question = sents[-1]
    # If the question has conditions, ignore them and find the surface
    if ',' in question:
      question = question.split(',')[-1]
    return question
  
  def addSchema(self, schema, schemas):
    qs = schema["quantities"]
    qSchemas = schema["quantity_schema"]
    for i, qSchema in sorted(qSchemas.items(), key=lambda x:int(x[0])):
      schemas['quantity_schema'].append(qSchema)
    
    schemas['quantities'] += qs

    return schemas

#  def extractQuantities(self):
#    quantities = []
#    for problem in self.problems:
#      aligns = problem['lAlignments']
#      sents = problem['']
#def dependencyParing():
#def posTagging():

if __name__ == "__main__":
  prep = Preprocessor("data/MultiArith.json", True)
  prep.prepare("data/multi_arith/problems_all.json") 
