"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import copy
from scores.func import *
from scores.FormulaTreePopul import FormulaTreePopul
from scores.FormulaPopulation import FormulaPopulation

def print_tree(tree):
    print(tree.to_strw())
    print(tree.to_str_oper())
    print(tree.to_str_connect())
    print("Complexity = {}".format(tree.complexity()))

def print_rank(fp, type_fit=None):
    for tree in fp.population:
        if type_fit == "mean-min":
            print("Formula={} Rank={} Target={} TargetMin={}".format(tree.to_strw(), tree.rank, tree.ftarget, tree.ftarget_min))
        elif type_fit == "mean-complexity":
            print("Formula={} Rank={} Target={} Complexity={}".format(tree.to_strw(), tree.rank, tree.ftarget, tree.complexity_target))
        else:
            print("Formula={} Target={} TargetMin={} Complexity={}".format(tree.to_strw(), tree.ftarget, tree.ftarget_min, tree.complexity_target))

if __name__ == "__main__":
    # Создание графа
    my_prob = np.ones(len(FUNC_OPERATIONS), dtype=int)
    tree2 = FormulaTreePopul(FUNC_OPERATIONS, ARGS_OPERATIONS, my_prob, 3, 3)
    tree = copy.deepcopy(tree2); tree.check_index()
    print_tree(tree)

    # Расчёт графа
    inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    targ = np.array([5, 7, 9])
    inp.shape = (inp.shape[0], 3)
    print("Input:", inp)
    print("Predict:", tree.predict(inp))

    # Изменение графа
    tree.rand_change_oper()
    print(); print("Change")
    print_tree(tree)

    # Расчёт графа
    print("Predict:", tree.predict(inp))

    # Добавление вершины
    tree.rand_add_vertex()
    print(); print("Add vertex 1")
    print_tree(tree)
    print(); print("Add vertex 2")
    tree2.rand_add_vertex()
    print_tree(tree2)

    # Расчёт графа
    print("Predict:", tree.predict(inp))

    # Кроссовер
    child = FormulaPopulation.crossover(tree, tree2)
    print(); print("Crossover")
    print_tree(child)

    # Создание популяции
    fp = FormulaPopulation(input_num=3, weights_num=3, my_func=FUNC_OPERATIONS, my_prob=my_prob, weights_popul=10,
                           prob_crossover=1.0, prob_func=1.0, prob_vertex=1.0)
    fp.init_popul(4)
    #print("Check set:", fp.check_set()); fp.check_index()
    #parents = fp.tournament(len(fp.population))
    fp.population[0] = tree
    fp.population[1] = tree2
    fp.multi_targetfun(inp, targ)
    print(); print("Targets")
    print_rank(fp)
    #print("Check set:", fp.check_set()); fp.check_index()

    # Ранжирование
    type_fit = fp.probMoo()
    fp.rank_sort()
    print(); print("Rank")
    print("TypeFit={}".format(type_fit))
    print_rank(fp, type_fit)
    #print("Check set:", fp.check_set()); fp.check_index()

    # Турнир
    children, nOffspring, numberToCull, nElites = fp.cull_elite(True)
    parents = fp.tournament(nOffspring)
    for ind in range(nOffspring):
        child = copy.deepcopy(fp.population[parents[0, ind]])
        children.append(child)
    fp.population = children
    print(); print("Tournament")
    print("Cull={} Elit={}".format(numberToCull, nElites))
    print_rank(fp)
    #print("Check set:", fp.check_set()); fp.check_index()

    # Запуск оптимизации
    print(); print("Run")
    for ind in range(10):
        #print("New iter:", ind, fp.check_set()); fp.check_index()
        fp.multi_targetfun(inp, targ)
        fp.probMoo()
        fp.population = fp.recombine()
    fp.sort(inp, targ)
    print_rank(fp)
