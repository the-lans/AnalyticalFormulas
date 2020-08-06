"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import copy
from scores.func import *
from scores.FormulaVertex import FormulaVertex
from scores.FormulaTree import FormulaTree
from scores.FormulaTreePopul import FormulaTreePopul
from scores.FormulaPopulation import FormulaPopulation

def print_tree(tree):
    print(tree.to_strw())
    print(tree.to_str_oper())
    print(tree.to_str_connect())
    print("Complexity = {}".format(tree.complexity()))

def print_rank_tree(tree, type_fit=None):
    if type_fit == "mean-min":
        print("Formula={} Rank={} Target={} TargetMin={}".format(tree.to_strw(), tree.rank, tree.ftarget, tree.ftarget_min))
    elif type_fit == "mean-complexity":
        print("Formula={} Rank={} Target={} Complexity={}".format(tree.to_strw(), tree.rank, tree.ftarget, tree.complexity_target))
    else:
        print("Formula={} Target={} TargetMin={} Complexity={}".format(tree.to_strw(), tree.ftarget, tree.ftarget_min, tree.complexity_target))

def print_rank(fp, type_fit=None):
    for tree in fp.population:
        print_rank_tree(tree)

def get_lin_tmodel(tree):
    tree.base = FormulaVertex(fsum)
    vertex = FormulaVertex(fmul)
    vertex.add(FormulaVertex(None, "x", 0))
    vertex.add(FormulaVertex(None, "w", 0))
    tree.base.add(vertex)
    tree.base.add(FormulaVertex(None, "w", 1))
    return tree

def get_lin_tmodel2(tree):
    tree.base = FormulaVertex(fsum)
    vertex1 = FormulaVertex(fmul)
    vertex2 = FormulaVertex(fmul)
    vertex2.add(FormulaVertex(None, "x", 0))
    vertex2.add(FormulaVertex(None, "w", 1))
    vertex1.add(vertex2)
    vertex1.add(FormulaVertex(None, "w", 0))
    tree.base.add(vertex1)
    tree.base.add(FormulaVertex(None, "w", 1))
    return tree

def get_model(weights, func):
    model = FormulaTree(weights=weights)
    model = func(model)
    model.update_index()
    return model

if __name__ == "__main__":
    my_prob = np.ones(len(FUNC_OPERATIONS), dtype=int)
    inp = np.array([[1], [2], [3], [4], [5]])
    targ = np.array([6, 9, 12, 15, 18]) #3 * x + 3
    inp.shape = (inp.shape[0], 1)
    print("Input:", inp)

    # Обучение единственного дерева
    tree = get_model(np.array([0.0, 0.0]), get_lin_tmodel2)
    tree.init_weights()
    print("predict =", tree.predict(inp))
    print("weights =", tree.weights)
    tree.fit(inp, targ, 100)
    print_tree(tree)
    print_rank_tree(tree)
    print("predict =", tree.predict(inp))
    print("weights =", tree.weights)

    # Обучение популяции
    print(); print("Formula Population:")
    fp = FormulaPopulation(input_num=1, weights_num=2, my_func=FUNC_OPERATIONS, my_prob=my_prob, weights_popul=20)
    fp.start_popul(30)
    #fp.start_popul_func(30, get_lin_tmodel)
    #fp.targetfit(inp, targ, maxiter=100)
    #fp.askfit(inp, targ, maxiter=100)
    fp.run(inp, targ, iter=100, iterfit=100)
    #fp.runfit(inp, targ, iter=100, maxiter=100)
    print_rank(fp)

    print(); print("Predict:")
    print(fp.predict(inp))

    print(); print("Weights:")
    print(fp.get_weights())
