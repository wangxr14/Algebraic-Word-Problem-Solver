import json
import numpy as np

#class Word2Vec():
  
# TODO: Add a function to filter out irrelevant quantities
class FeatureGenerator:
  def __init__(self, problemFile, debug=False):
    with open(problemFile, 'r') as f:
      self.problems = json.load(f)
    self.schema = {qid: schema["quantity_schema"] for qid, schema in self.problems.items()}
    self.questions = {qid: schema["question"] for qid, schema in self.problems.items()}
    self.relevanceFeatures = {}
    self.lcaFeatures = {}
    self.debug = debug

  def extractFeatures(self):
    for qid, schema, question in zip(self.schema.keys(), self.schema.values(), self.questions.values()):
      if self.debug:
        print(qid, schema.keys(), question)
      self.relevanceFeatures[qid] = self.extractRelevanceFeature(schema, question)
      self.lcaFeatures[qid] = self.extractLcaFeature(schema, question)

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
        
        lcaFeatures[' '.join([q1, q2])] = feat

    return lcaFeatures

if __name__ == "__main__":
  problemFile = "data/lca_solver_test/test.json"
  relevFeatFile = "data/lca_solver_test/relev_features.json"
  lcaFeatFile = "data/lca_solver_test/lca_features.json"
  
  # Create a fake problem file
  problems = {}
  problem = {"quantity_schema":{'3':{}, '5':{}, '4':{}}}
  problem["quantity_schema"]['3'] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":"candies", 
                  "noun_phrases": ["store", "chocalates"]} 

  problem["quantity_schema"]['5'] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":"candies", 
                  "noun_phrases":["store", "brownies"]} 

  problem["quantity_schema"]['4'] = {"verb":"sell",
                  "subject":"Mary",
                  "unit":"candies", 
                  "noun_phrases":["home", "chocalates"]} 
  
  problem["question"] = "How many candies does John have ?"

  problems['0'] = problem
  with open("data/lca_solver_test/test.json", 'w') as f:
    json.dump(problems, f)

  fgen = FeatureGenerator(problemFile, debug=True)
  fgen.extractFeatures()
  with open(relevFeatFile, 'w') as f:
    json.dump(fgen.relevanceFeatures, f)

  with open(lcaFeatFile, 'w') as f:
    json.dump(fgen.lcaFeatures, f)
