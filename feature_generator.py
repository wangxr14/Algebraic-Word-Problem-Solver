import json
import numpy as np
import nltk
from label_generator import *
from utils import *

#TODO: Rename self.problem to self.schemas
#class Word2Vec():
class Unigram:
  def __init__(self,
              rawProblemFile,
              debug=False):
    self.debug = debug
    with open(rawProblemFile, 'r') as f:
      self.problems = json.load(f)
    
    self.corpus = []
    for i, p in enumerate(self.problems):
      self.corpus.append([])
      p_text = p['sQuestion']
      
      # TODO: Implement tokenization; this may not work if the numbers contain decimal point
      sents_raw = nltk.word_tokenize(p_text) 
      self.corpus[i] = sents_raw

    self.word2id = {}
    count = 0
    for sents in self.corpus:
      for w in sents:
        if w not in self.word2id:
          self.word2id[w] = count
          count += 1
    self.word2id['UNK'] = count
    self.nvocab = count + 1 
    print("Total number of vocabs:", count)

  def extract(self, sentence):
    v = np.zeros((self.nvocab,))
    for w in sentence:
      if w in self.word2id:
        v[self.word2id[w]] = 1.
      else:
        v[self.word2id['UNK']] = 1.        

    return v.tolist()

# TODO: Add a function to filter out irrelevant quantities
class FeatureGenerator:
  def __init__(self,
              schemaFile, 
              problemFile, 
              equationFile, 
              mathGrammarFile='mathGrammar.pcfg',
              debug=False):
    self.debug = debug
    with open(schemaFile, 'r') as f:
      self.problems = json.load(f)
    
    self.unigram = Unigram(problemFile)

    self.lcaLabelGen = LcaLabelGenerator(mathGrammarFile, debug=debug)
    self.lcaLabels = self.lcaLabelGen.generateFromFile(equationFile)
    
    self.relevanceFeatures = [[] for i in range(len(self.problems))]
    self.lcaFeatures = [[] for i in range(len(self.problems))]
    self.relevanceLabels = [[] for i in range(len(self.problems))]
    self.quantities = [[] for i in range(len(self.problems))]
  
  def extractFeatures(self, test=False):
    for pid in range(len(self.problems)):
      print('Problem', pid)
      self.quantities[pid] = self.problems[pid]['quantities']
      question = self.problems[pid]['question']
      schema = self.problems[pid]['quantity_schema']
      p_text = self.unigram.corpus[pid]
      if self.debug:
        print(self.problems[pid]['quantity_schema'])
      
      #self.relevanceFeatures[pid] = self.extractRelevanceFeatures(schema, question)
      self.lcaFeatures[pid] = self.extractLcaFeatures(schema, question, p_text)
    
  def extractRelevanceFeatures(self, schema, question):
    relevanceFeatures = {}
    for q in range(len(schema)):
      if self.debug:
        print(schema.keys()) 
      
      q_schema = schema[q]
      feat = {} #np.zeros((6,))
      
      verb = q_schema["verb"]
      subject = q_schema["subject"]
      units = q_schema["unit"]
      nps = q_schema["noun_phrases"] + units + [subject]
      
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

      for q2 in range(len(schema)):
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

  def extractLcaFeatures(self, 
                        schema, 
                        question,
                        p_text, 
                        use_individual=True):
    lcaFeatures = {}
    vs = []
    if use_individual:
      # Find the unigram feature for each quantity with neighboring 
      # adjective and adverbs as context
      q_contexts = self.extractQuantityContext(p_text)
        
      for q in range(len(schema)):
        q_schema = schema[q]
        verb = [q_schema['verb'].split('_')[-1]]
        v = self.unigram.extract(verb + q_contexts[q])
        vs.append(v)

    for q1 in range(len(schema)):
      for q2 in range(len(schema)):
        if self.debug:
          print('Test loop')
        if q1 == q2:
          if self.debug:
            print('Should not go further')
          continue
        if self.debug:
          print('Go further')
        q_schema1 = schema[q1]
        q_schema2 = schema[q2]

        feat = {} #np.zeros((5,))
      
        verb1, verb2 = q_schema1["verb"], q_schema2["verb"]
        subject1, subject2 = q_schema1["subject"], q_schema2["subject"]
        units1, units2 = q_schema1["unit"], q_schema2["unit"]
        rate1, rate2 = q_schema1["rate"], q_schema2["rate"]
        nps1, nps2 = q_schema1["noun_phrases"], q_schema2["noun_phrases"]
        
        feat['unigram_1'] = vs[q1]
        feat['unigram_2'] = vs[q2]  
        if verb1 == verb2:
          feat['verb_match'] = 1.
        else:
          feat['verb_match'] = 0.

        if len(rate1):
          feat['is_rate_1'] = 1.
        else:
          feat['is_rate_1'] = 0.

        if len(rate2):
          feat['is_rate_2'] = 1.
        else:
          feat['is_rate_2'] = 0.
  
        if units1 == units2:
          feat['unit_match'] = 1.
        else:
          feat['unit_match'] = 0.

        # If one of the quantity is a rate, determine which component of
        # it matches the other quantity's unit
        flag = 0
        if feat['is_rate_1'] or feat['is_rate_2']:
          for k, unit1 in enumerate(units1):
            for l, unit2 in enumerate(units2):
              if unit1 == unit2:
                feat['unit_match_pos_1'] = k
                feat['unit_match_pos_2'] = l
                flag = 1
            
        if not flag:
          feat['unit_match_pos_1'] = -1
          feat['unit_match_pos_2'] = -1


        # Check if the first value is larger than the second one; useful
        # for detecting reverse operation
        #if float(q1) > float(q2):
        #  feat['q1_greater'] = 1.
        #else:
        #  feat['q1_greater'] = 0.

        # Check if "more", "less" and "than" appear in the question,
        # as they are useful to know the kind of operations used in the 
        # question
        feat['compare_words_in_question'] = 0.
        for w in ["more", "less", "than"]:
          if w in question:
            feat['compare_words_in_question'] = 1.
            break
        
        lcaFeatures['_'.join([str(q1), str(q2)])] = feat

    return lcaFeatures   

  def extractQuantityContext(self, p_text, window_size=5, filter_tags=['JJ', 'JJR', 'RBR']):
    width = int((window_size - 1) / 2)
    quantity_contexts = []
    if self.debug:
      print(p_text)
    tags = nltk.pos_tag(p_text)

    # Only use the tag labels
    tags = [tag[1] for tag in tags]
    len_sent = len(tags)

    for i, w in enumerate(p_text):
      right_width = min(len_sent - i - 1, width)
      left_width = min(i, width)
    
      if isQuantity(w):
        quantity_context = []
        context = p_text[i-left_width:i+right_width]
        if self.debug:
          print(w, context, tags[i-left_width:i+right_width])
        for j, cw in enumerate(context):
          cur_pos = i + j - left_width
          if tags[cur_pos] in filter_tags:
            quantity_context.append(cw)

        quantity_contexts.append(quantity_context)
    if self.debug:
      print(quantity_contexts)
    return quantity_contexts

  def save(self, featFile):
    features = {}
    features['lca_features'] = self.lcaFeatures
    features['lca_labels'] = self.lcaLabels['labels'] 
    features['rel_features'] = self.relevanceFeatures
    #features['rel_labels'] = self.relevanceLabels 
    features['quantities'] = self.quantities
    features['math_ops'] = self.lcaLabels['mathOps']
  
    with open(featFile, 'w') as f:
      json.dump(features, f, indent=4, sort_keys=True)

if __name__ == "__main__":
  schemaFile = "data/multi_arith/problems_all.json"
  problemFile = "data/MultiArith.json"
  equationFile = "data/MultiArith.json"
  relevFeatFile = "data/multi_arith/relev_features.json"
  lcaFeatFile = "data/multi_arith/lca_features.json"
  grammarFile = "data/lca_solver_test/mathGrammar.pcfg"
  
  # Create a fake problem file
  '''problems = []
  problem = {"quantity_schema":[{}, {}, {}]}
  problem["quantity_schema"][0] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":["candies", "per", "bag"],
                  "rate":["bag"], 
                  "noun_phrases": ["store", "chocalates"]} 

  problem["quantity_schema"][1] = {"verb":"buy", 
                  "subject":"John", 
                  "unit":["candies", "per", "bag"],
                  "rate":["bag"], 
                  "noun_phrases":["store", "brownies"]} 

  problem["quantity_schema"][2] = {"verb":"sell",
                  "subject":"Mary",
                  "unit":["candies", "per", "box"],
                  "rate":["box"], 
                  "noun_phrases":["home", "chocalates"]} 
  
  problem["question"] = "How many candies does John have ?"
  problem["quantities"] = ['3', '5', '4']
  problems = [problem]*6

  equations = [{'iIndex':'0', "lEquations": ["X=((3.0*5.0)/4.0)"]}]*6
   
  with open(schemaFile, 'w') as f:
    json.dump(problems, f)
  
  with open(equationFile, 'w') as f:
    json.dump(equations, f)
  '''
  fgen = FeatureGenerator(schemaFile, problemFile, equationFile, grammarFile, debug=True)
  fgen.extractFeatures()
  fgen.save("data/multi_arith/features.json")
