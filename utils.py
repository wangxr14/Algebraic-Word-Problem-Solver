import nltk

def isQuantity(w):  
  for c in w:
    if c > '9' or c < '0':
      # Check if it is a floating point value or fraction
      if c != '.':
        return 0
  return 1
      
def isMathOp(w, mathOps):
  if w in mathOps:
    return 1
  return 0
  
def tokenizeEq(equation):
  tokens = []
  nchar = len(equation)
  i = 0
  while i < nchar:
    if isQuantity(equation[i]):
      j = i
      while isQuantity(equation[i:j+1]):
        j += 1
      tokens.append(equation[i:j])
      i = j 
    else:
      tokens.append(equation[i])
      i += 1
  return tokens
  
def tokenizeProblem(p_text):
  tokens = nltk.pos_tag(p_text)
  tokenized_problem = []
  cur_sent = []
    
  for t in tokens:
    if t != '.':
      cur_sent.append(t)
    else:
      tokenized_problem.append(cur_sent)
      cur_sent = []
  return tokenized_problem
