"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import copy
import time
import csv
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

def csv_rank(path, fp):
    with open(path, "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ['Formula', 'Rank', 'Target', 'TargetMin', 'Complexity'], delimiter=';')
        writer.writeheader()
        for tree in fp.population:
            line = {'Formula': tree.to_strw(), 'Rank': tree.rank, 'Target': tree.ftarget, 'TargetMin': tree.ftarget_min, 'Complexity': tree.complexity_target}
            writer.writerow(line)

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

def write_file(name, data):
    with open(name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for line in data:
            writer.writerow(line)

def func(start, stop, num=30, disp=0.0):
    x = np.linspace(start, stop, num)
    y = 2 * x**2 - 7 * x - 5
    return (x, np.random.normal(y, disp, (1, x.shape[0])).flatten())

if __name__ == "__main__":
    #my_prob = [1, 1, 1, 1, 1, 1, 1]
    my_prob = np.ones(len(FUNC_OPERATIONS), dtype=int)
    inp, targ = func(-3, +7, 35, 2.0)
    write_file('data.csv', np.hstack((inp.reshape(inp.shape[0], 1), targ.reshape(targ.shape[0], 1))))
    inp.shape = (inp.shape[0], 1)
    #print("Input:", inp)

    # Обучение популяции
    print(); print("Start fit:")
    fp = FormulaPopulation(input_num=1, weights_num=2, my_func=FUNC_OPERATIONS, my_prob=my_prob, weights_popul=20)
    fp.is_print_iter = True
    fp.start_popul(30)
    #fp.start_popul_func(30, get_lin_tmodel)
    start_time = time.process_time()
    #fp.targetfit(inp, targ, maxiter=100)
    #fp.askfit(inp, targ, maxiter=100)
    #fp.run(inp, targ, iter=100, iterfit=100)
    fp.runfit(inp, targ, iterfit=20, iter=100)
    fp.runfit(inp, targ, iterfit=100)
    print("Time = {:g} sec".format(time.process_time() - start_time))

    print(); print("Formula Population:")
    print_rank(fp)
    csv_rank('population.csv', fp)

    print(); print("Predict:")
    write_file('predict.csv', fp.predict(inp).T)

    print(); print("Weights:")
    print(fp.get_weights())
    write_file('weights.csv', fp.get_weights())
