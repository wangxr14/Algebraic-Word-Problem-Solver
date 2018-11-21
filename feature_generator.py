import json
import numpy as np
from label_generator import *
from utils import *
#class Word2Vec():
  
# TODO: Add a function to filter out irrelevant quantities
class FeatureGenerator:
  def __init__(self, 
              problemFile, 
              equationFile, 
              mathGrammarFile='mathGrammar.pcfg',
              debug=False):
    with open(problemFile, 'r') as f:
      self.problems = json.load(f)
    with open(equationFile, 'r') as f:
      self.equations = json.load(f)
    
    self.lcaLabelGen = LcaLabelGenerator(mathGrammarFile)
    self.relevanceFeatures = {}
    self.lcaFeatures = {}
    self.lcaLabels = {}
    self.relevanceLabels = {}
    self.quantities = {}
    self.debug = debug

  def extractFeatures(self, test=False):
    for pid in self.problems.keys():
      self.quantities[pid] = self.problems[pid]['quantities']
      question = self.problems[pid]['question']
      schema = self.problems[pid]['quantity_schema']
      equation = self.equations[pid]['lEquations']
      if self.debug:
          print(problems[pid]['quantity_schema'].values())
      
      self.relevanceFeatures[pid] = self.extractRelevanceFeature(schema, question)
      self.lcaFeatures[pid] = self.extractLcaFeature(schema, question)
      if test:
        self.lcaLabels = {'0':{'0_1': 1, '0_2': 2, '1_2': 2, '1_0': 1, '2_0': 3, '2_1': 3}}
      else:
        self.lcaLabels[pid] = self.extractLcaLabels(equation) 
  
  def extractLcaLabels(self, equation):
    expression = equation.split('=')[-1]
    tokens = tokenizeEq(expression)
    labels = self.lcaLabelGen.generate(tokens)
    return labels

  def extractRelevanceFeature(self, schema, question):
    relevanceFeatures = {}
    for q in schema.keys():
      q_schema = schema[q]
      if self.debug:
        print(schema.keys()) 
      feat = {} #np.zeros((6,))
      
      verb = q_schema["verb"]
      subject = q_schema["subject"]
      units = q_schema["unit"]
      nps = q_schema["noun_phrases"]
      
      # Find if the unit of the quantity appears in the question
      countUnits = self.countPatternsInQuestion(units, question)
      if countUnits > 0:
        feat['unit_in_question'] = 1.
      else:
        feat['unit_in_question'] = 0.

      # Find the number of related NPs for the quantity
      countNps = self.countPatternsInQuestion(nps, question)
      if countNps > 0:
        feat['np_in_question'] = 1.
      else:
        feat['np_in_question'] = 0.

      feat['q_with_more_units'] = 0.
      feat['n_q_with_more_units'] = 0.
      feat['n_q'] = 0.

      for q2 in schema.keys():
        # Search all quantities to see if any other quantities have more units
        # than the current unit and if so, how many quantities
        countUnits2 = self.countPatternsInQuestion(units, question)
        if countUnits2 > countUnits:
          feat['q_with_more_units'] = 1.
          feat['n_q_with_more_units'] += 1.
      
        # Search all quantities to see if any other quantities have more related NPs
        # than the current quantity in the question
        countNps2 = self.countPatternsInQuestion(nps, question)
        if countNps2 > countNps:
          feat['q_with_more_nps'] = 1.
        else:
          feat['q with more nps'] = 0.

        # Find the number of quantities in the text (fewer quantities will
        # be more likely to be relevant)
        feat['n_q'] += 1.
      
      relevanceFeatures[q] = feat
    return relevanceFeatures    

  def countPatternsInQuestion(self, patterns, question):
    count = 0
    for pattern in patterns:
      if pattern in question:
        count += 1
    return count

  def extractLcaFeature(self, schema, question):
    lcaFeatures = {}
    for q1 in schema.keys():
      for q2 in schema.keys():
        if q1 == q2:
          continue
        q_schema1 = schema[q1]
        q_schema2 = schema[q2]

        feat = {} #np.zeros((5,))
      
        verb1, verb2 = q_schema1["verb"], q_schema2["verb"]
        subject1, subject2 = q_schema1["subject"], q_schema2["subject"]
        units1, units2 = q_schema1["unit"], q_schema2["unit"]
        nps1, nps2 = q_schema1["noun_phrases"], q_schema2["noun_phrases"]
      
        if verb1 == verb2:
          feat['verb_match'] = 1.
        else:
          feat['verb_match'] = 0.

        if len(units1) >= 2:
          feat['is_rate_1'] = 1.
        else:
          feat['is_rate_1'] = 0.

        if len(units2) >= 2:
          feat['is_rate_2'] = 1.
        else:
          feat['is_rate_2'] = 0.
  
        flag = 0
        for i, u1 in enumerate(units1):
          for j, u2 in enumerate(units2):
            if u1 == u2:
              flag = 1
              feat['unit_match'] = 1.

              # Determine if one of the quantity is a rate (associated
              # with two units)
              if len(units1) >= 2: 
                feat['unit_match_pos'] = i
              elif len(units2) >= 2:
                feat['unit_match_pos'] = j
              
        if not flag:
          feat['unit_match_pos'] = -1

        # Check if the first value is larger than the second one; useful
        # for detecting reverse operation
        if float(q1) > float(q2):
          feat['q1_greater'] = 1.
        else:
          feat['q1_greater'] = 0.

        # Check if "more", "less" and "than" appear in the question,
        # as they are useful to know the kind of operations used in the 
        # question
        feat['compare_words_in_question'] = 0.
        for w in ["more", "less", "than"]:
          if w in question:
            feat['compare_words_in_question'] = 1.
            break
        
        lcaFeatures['_'.join([q1, q2])] = feat

    return lcaFeatures

  def save(self, featFile):
    features = {}
    features['lca_features'] = self.lcaFeatures
    features['lca_labels'] = self.lcaLabels 
    features['rel_features'] = self.relevanceFeatures
    #features['rel_labels'] = self.relevanceLabels 
    features['quantities'] = self.quantities

    with open(featFile, 'w') as f:
      json.dump(features, f)

if __name__ == "__main__":
  problemFile = "data/lca_solver_test/test_problem.json"
  equationFile = "data/lca_solver_test/test_equation.json"
  relevFeatFile = "data/lca_solver_test/relev_features.json"
  lcaFeatFile = "data/lca_solver_test/lca_features.json"
  
  # Create a fake problem file
  problems = {}
  problem = {"quantity_schema":{0:{}, 1:{}, 2:{}}}
  problem["quantity_schema"][0] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":"candies", 
                  "noun_phrases": ["store", "chocalates"]} 

  problem["quantity_schema"][1] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":"candies", 
                  "noun_phrases":["store", "brownies"]} 

  problem["quantity_schema"][2] = {"verb":"sell",
                  "subject":"Mary",
                  "unit":"candies", 
                  "noun_phrases":["home", "chocalates"]} 
  
  problem["question"] = "How many candies does John have ?"
  problem["quantities"] = ['3', '5', '4']
  problems['0'] = problem
  
  equations = {'0':{"lEquations": "X=((3.0+5.0)-4.0)"}}
   
  with open(problemFile, 'w') as f:
    json.dump(problems, f)
  
  with open(equationFile, 'w') as f:
    json.dump(equations, f)

  fgen = FeatureGenerator(problemFile, equationFile, debug=True)
  fgen.extractFeatures(test=True)
  fgen.save("data/lca_solver_test/test_features.json")
