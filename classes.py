from collections import deque

DEBUG = True
class Item:

    def __init__(self, label, prob, numParses):
        self.label = label
        self.prob = prob
        self.numParses = numParses

    # Returns the node's label
    def toString(self):
        return self.label


class Node(Item):
    def __init__(self, category, score, children=()):
        Item.__init__(self, category, score, 0)
        self.children = children
        self.score = score
        self.leafIds = []
        # Keep a list of indices for the quantities in the subtree
        if len(children) == 0:
          self.leafIds = [category]
        else:
          for c in children:
            self.leafIds += c.leafIds 
        

        if len(children) == 1:
          if category == TOP:
            self.numParses = children[0].numParses
          else:
            self.numParses = 1
        elif len(children) == 2:
          self.numParses = children[0].numParses * children[1].numParses
        elif len(children) == 0:
          self.numParses = -1  # dummy numParses value; this should not be -1!
        
        elif len(self.children) > 2:
            print("Warning: adding a node with more than two children (CKY may not work correctly)")

    # 
    # Find all the LCA operations in the expression tree
    #
    def findLcas(self, mathOps=['+', '-', '*', '/', '-_rev', '/_rev']):
      queue = deque([self])
      pairToLcas = {}
      while len(queue): 
        node = queue.popleft()
        children = node.children
        if DEBUG:
          print(children)
        	
        if node.label == "TOP":
          if DEBUG:
            print(children[0].label)
          queue.append(children[0])  
          #pairtolcas = findLcas(children[0], pairToLcas)
        if len(children) == 3:
          # Check the label of the middle child, which may contain the operation
          # type between the two subtrees
          opLabel = children[1].label
          if opLabel in mathOps:
            lChild = children[0]
            rChild = children[2]
            if DEBUG:
              print(lChild.label, children[1].label, rChild.label)
            for ql in findLeafIds(lChild):
              for qr in findLeafIds(rChild):
                if DEBUG:
                  print(ql, qr)
                pairToLcas['_'.join([ql, qr])] = opLabel
                # For noncommutative operations, add reverse operation labels
                if opLabel == '-' or opLabel == '/':
                  pairToLcas['_'.join([qr, ql])] = opLabel + '_rev'
                else:
                  pairToLcas['_'.join([qr, ql])] = opLabel

            queue.append(lChild)
            queue.append(rChild)
        
          else:
            mChild = children[1]
            queue.append(mChild)
    
        return pairToLcas


    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"

if __name__ == "__main__":
  a = Node('1', 1.0)
  b = Node('2', 1.0)
  c = Node('+', 1.0, (a, b))
  print(c.findLcas())
