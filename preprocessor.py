import json
import os
from nltk.tree import ParentedTree
from nltk.parse import stanford
from nltk.tokenize import sent_tokenize
from quantity_schema2 import *
from utils import *

# TODO: pretty print in json
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
      json.dump(prblmDict, f, indent=4, sort_keys=True)
  
  def extractQuantitySchema(self):
    qSchemas = []
    for i, problem in enumerate(self.problems):
      print('Problem', i)
      # TODO: Use the tokenizer instead: can be a problem when dealing with
      # floating point values
      sents = nltk.sent_tokenize(problem['sQuestion'])
      pSents = self.pcfgParser.raw_parse_sents(sents)
      dSents = self.depParser.raw_parse_sents(sents)
      
      schemas = {'quantities':[], 'quantity_schema':[]}
      for j, (pline, dline) in enumerate(zip(pSents, dSents)):
        for psent, dsent in zip(pline, dline):
          pcfgTree = ParentedTree.fromstring(str(psent)) 
          depTree = dsent 
          schema = dealWithSentence(pcfgTree, depTree)
          # Label each verb with the sentence order
          for k, q_schema in enumerate(schema['quantity_schema']):
            q_schema['verb'] = str(j) + '_' + q_schema['verb']
            schemas['quantity_schema'].append(q_schema)
          
          schemas['quantities'] += schema['quantities']
          
        n = len(schemas['quantities']) 
        # Compensate for the missing units
        for k in range(n):
          if not schemas['quantity_schema'][k]['unit']:
              found = 0
               
              for l in range(k):
                if self.debug:
                  print('Detected missing unit, replaced with: ', schemas['quantity_schema'][k-l-1]['unit'])
               
                if schemas['quantity_schema'][k-l-1]['unit']:
                  schemas['quantity_schema'][k]['unit'] = schemas['quantity_schema'][k-l-1]['unit']
                  found = 1
                  break
              if found:
                if self.debug:
                  print('Found missing unit')
                continue
              else:
                for l in range(n - k):
                  if schemas['quantity_schema'][k+l]['unit']:
                    schemas['quantity_schema'][k]['unit'] = schemas['quantity_schema'][k+l]['unit']
                    break
 
      schemas['question'] = self.findQuestion(sents)

      #if self.debug:
      quantities = schemas['quantities']
      print(quantities, schemas)
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
    for i, qSchema in enumerate(qSchemas):      
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
  prep = Preprocessor("data/add_sub/AddSub.json", True)
  prep.prepare("data/add_sub/schema_all_test.json") 
