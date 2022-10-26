from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import pandas as pd

def generate_rules(X, ite, n_trees=1, max_depth=3, digits=2):
    for _ in range(n_trees):
        model = DecisionTreeRegressor(max_depth = max_depth)
        model.fit(X, ite)
        #print(tree.export_text(model))
        rules = get_rules(model, X.columns, digits)
    return rules


def get_rules(tree, feature_names, digits=2, min_cases=0):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"(X['{name}']<={np.round(threshold, digits)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"(X['{name}']>{np.round(threshold, digits)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [tree_.n_node_samples[node]]
            paths += [path]
            
    recurse(0, path, paths)
    
    rules = []
    for path in paths:

        # TO DO: discard too specific rules
        if path[-1]<min_cases: break
        path = path[:-1]

        # clean path
        path.sort()
        for feature_name in feature_names:
            path_i = [node for node in path if feature_name in node]
            if len(path_i)>1:
                path_i_magg = [node for node in path_i if ">" in node]
                if path_i_magg is not None: 
                    if len(path_i_magg)>1:
                        for redundant in path_i_magg[:-1]: 
                            path.remove(redundant)
                path_i_min = [node for node in path_i if "<" in node]
                if path_i_min is not None:
                    if len(path_i_min)>1:
                        for redundant in path_i_min[1:]: 
                            path.remove(redundant)
        rule = ""  
        for p in path:
            if rule != "":
                rule += " & "
            rule += p
        rules += [rule]
        
    return rules


def get_rules_matrix(rules, X):
    R = {}
    for rule in rules:
        R[rule] = eval(rule).astype(int)
    return pd.DataFrame(R)


def rules_regularization(R, t_anom=0.02):
    # disard too genaric or too rare rules
    generic_rules = R.describe().loc["mean"]>1-t_anom
    rare_rules = R.describe().loc["mean"]<t_anom
    R = R.loc[:,~(generic_rules | rare_rules)]
    return R.columns