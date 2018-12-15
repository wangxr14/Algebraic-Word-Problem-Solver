# Algebraic-Word-Problem-Solver

#### Description:

This is a python implementation of the following work:
~~~~
  Subhro Roy and Dan Roth.  
  Solving General Arithmetic Word Problems.  
  EMNLP 2015.
~~~~

#### Data:

The data are from http://lang.ee.washington.edu/MAWPS/ developed by:
~~~~
  Rik Koncel-Kedziorski, Subhro Roy, Aida Amini,
Nate Kushman, and Hannaneh Hajishirzi.
  MAWPS: A Math Word Problem Repository.
  NAACL 2016.
~~~~
The data are under the directory ./data/. We support two
datasets for now: MultiArith.json and SingleOp.json.

#### How to run it
cd to the repo and run:
~~~~
python main.py --dset [dataset name]
~~~~
The results will be store in data/multi_arith if MultiArith dataset
is used and data/single_op if SingleOp is used. The model used 
crossvalidation so there will be n different result files with the
following filename:
~~~~
  pred_lcas_*.json: Store the predicted LCA labels
  pred_tree_*.json: Store the predicted expression trees
  gold_lcas_*.json: Store the correct LCA labels
~~~~

#### Advanced Options
To change the type of features used, go into main.py as modify the following code:
~~~~
  feat_choices = {...}
~~~~
To change the type of constraint used, modify the following in main.py:
~~~~
  constraints = {...}
~~~~
To generate quantity schema, modify the dataset path in preprocessor.py and run:
~~~~
  python preprocessor.py
~~~~

#### Questions
Feel free to email lwang114@illinois.edu and xinranw5@illinois.edu for any further questions
