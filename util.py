
# Can use isnumeric() instead?
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
