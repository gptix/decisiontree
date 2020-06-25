"""Implements class DecisionTreeclassifier, to 
- hold input data (attributes, classes), 
- fit a binary decision tree to efficiently decide a predicted class 
for an input row of data
- make predictions

Attributes:
rows
colnames (not currently used)
tree (added by fit() method)

Methods:
fit()
predict()

Major helper functions defined:
make_tree()
gini_impurity()
imformation_gain()

Uses recursion to build a tree from input data rows.
Uses Gini Impurity and Information Gain functions to find optimal 
evolution of binary tree.

"""



class DecisionTree:
    def __init__(self, rows, colnames=None):
        self.rows = rows
        if colnames is None:
            colnames = ['column_' + str(i) for i in range(len(rows[0]))]
            
    def fit(self):
        self.tree = build_tree(self.rows)
        
    def predict(self, data):
        """Wrapper method to run 'predict_inner.'"""
        # TODO: unify classes 'DecisionTree' and 'DecisionNode'
        # They are basically the same thing.
        node = self.tree
        return predict_inner(data, node)
    
    
def predict_inner(row, node):
    """Helper function to be called by wrapper method 'predict'."""

    # Base case: Leaf node.
    if isinstance(node, LeafNode):
        return node.predictions

    # Recurse case.
    # Use this node's tester to choose a side to recurse down.
    if node.tester.passes(row):
        return predict_inner(row, node.true_side)
    else:
        return predict_inner(row, node.false_side)
    
    
class DecisionNode:
    """Holds a test to be applied, and pointers to two child nodes.
    Each child node can be a further DecisionNode, or a LeafNode.
    """

    def __init__(self, tester, true_side, false_side):
        self.tester = tester
        self.true_side = true_side
        self.false_side = false_side


class LeafNode:
    """Holds info about features.
    
    A single entry dict of feature value (classification) and 
    the count of times the feature value appears in the overall 
    input set.
    """

    def __init__(self, rows):
        self.predictions = tally(rows)

def distinct(rows, col):
    """Return the set of values from a specific column in a matrix."""
    return set([row[col] for row in rows])

def tally(rows):
    """Counts occurrences of specific values. Returns a 
    dict of    label: count    pairs."""
    tal = {} 
    for row in rows:
        # label is rightmost column
        label = row[-1]
        if label not in tal:
            tal[label] = 0 # add an entry for a new label
        tal[label] += 1
    return tal

class Tester:
    """A test to choose to which of two lists a tested row should be added."""
    
    def __init__(self, col, val):
        self.col = col # index of a column in a matrix.  Identifies a 'variable'
        self.val = val # the value of a 'variable'
        
    def passes(self, test_case):
        # Compare the feature value to a test value.
        test_val = test_case[self.col]
        if isinstance(self.val, int) or isinstance(self.val,float):
            
            # use greater than or equal for numeric values
            return test_val >= self.val
        else:
            
            # use double equals for string values
            return test_val == self.val
        

    def __repr__(self):
        # Print the actual test being applied.
        if isinstance(self.val, int) or isinstance(self.val,float):
            test = ">="
        else:
            test = "=="
        return f"Test whether {str(self.val)} matches column {self.col}, using {test}"


def split(data_rows, test):
    """Divides a data set into two 'child' datasets, using a test.
    
    For each row, if the test returns True, that row will be added to 
    the list 'true_rows'. If the test does not return True, the row is
    added to 'false rows'.
    """
    true_rows = []
    false_rows = []
    for row_to_test in data_rows:
        if test.passes(row_to_test):
            true_rows.append(row_to_test)
        else:
            false_rows.append(row_to_test)
            
    return true_rows, false_rows


def gini_impurity(rows):
    """Calculate the Gini impurity for a set of values in rows.
    """
    label_counts = tally(rows)
    
    impurity = 1 # start with complete impurity
    
    # adjust impurity for each label
    for label in label_counts:
        label_probability = label_counts[label] / float(len(rows))
        impurity -= label_probability**2 
        
    return impurity


def information_gain(left_child, right_child, uncertainty):
    """Information Gain.

    Defined as the result of subtracting the WEIGHTED impurities of two 
    CHILDREN from UNCERTAINTY of the CURRENT node.
    """
    prob = float(len(left_child)) / (len(left_child) + len(right_child))
    
    ig = uncertainty - prob * gini_impurity(left_child) - (1 - prob) * gini_impurity(right_child)
    return ig


def best_split(rows):
    """Given a set of observations, 
    - iterate over each pair of feature and value.
      - Calculate information gain.
      - Retain best pair each iteration.
    - Return final best pair."""
    
    # Initialize variables to track best gain and
    # the test question used to get that gain.
    best_gain = 0 
    best_test_q = None  # keep train of the feature / value that produced it
    
    uncertainty = gini_impurity(rows)
    col_count = len(rows[0]) - 1  

    for col in range(col_count):  # iterate over features

        distinct_vals = set([row[col] for row in rows])

        for val in distinct_vals:

            test_question = Tester(col, val)

            # Try to split rows into subsets
            true_rows, false_rows = split(rows, test_question)

            # If there is no split, ignore.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate information gain
            gain = information_gain(true_rows, false_rows, uncertainty)

            if gain > best_gain:
                best_gain, best_test_q = gain, test_question

    return best_gain, best_test_q

def build_tree(rows):
    """Recursively builds a decision tree to classify entries in the input rows.
    DecisionNodes hold test questions and subtrees.
    LeafNodes hold feature classifications and counts of those.
    """

    # Split the data (if possible).
    # find information gain per possible tests.
    # get best test.
    gain, tester = best_split(rows)

    # Base case: stop recursing, because we cannot gain 
    # more info by testing.
    # Create a LeafNode.
    if gain == 0:
        return LeafNode(rows)

    # Recursive case.  We found a test that delivers info gain.
    # Extend tree.
    true_rows, false_rows = split(rows, tester)
    true_subtree = build_tree(true_rows)
    false_subtree = build_tree(false_rows)

    # Build DecisionNode using subtrees just built, and the best test found.
    return DecisionNode(tester, true_subtree, false_subtree)


def print_tree(node, spacing=""):
    """Print out tree structure."""

    # Base case: Leaf node
    if isinstance(node, LeafNode):
        print (spacing + "Prediction: ", node.predictions)
        return

    print (spacing + str(node.tester))

    # Recurse down true branch
    print (spacing + '--> True:')
    print_tree(node.true_side, spacing + "  ")

    # Recurse down false branch
    print (spacing + '--> False:')
    print_tree(node.false_side, spacing + "  ")


# training_data = [
#     ['Green', 3, 'Apple'],
#     ['Yellow', 3, 'Apple'],
#     ['Red', 1, 'Grape'],
#     ['Red', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon'],
# ]


# # In[15]:


# too = DecisionTree(training_data)
# too.fit()
# too.predict(training_data[0])

