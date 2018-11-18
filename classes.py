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

    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"
