"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
from scores.func import *
from scores.FormulaVertex import FormulaVertex
from scores.FormulaTree import FormulaTree

def get_lin_model(weights):
    model = FormulaTree(weights=weights)
    model.base = FormulaVertex(fsum)
    vertex = FormulaVertex(fmul)
    vertex.add(FormulaVertex(None, "x", 0))
    vertex.add(FormulaVertex(None, "w", 0))
    model.base.add(vertex)
    model.base.add(FormulaVertex(None, "w", 1))
    model.update_index()
    return model

def get_lin_model2():
    model = FormulaTree()
    model.base = FormulaVertex(fsum)
    vertex = FormulaVertex(fdev)
    vertex.add(FormulaVertex(None, "x", 0))
    vertex.add(FormulaVertex(None, "x", 0))
    model.base.add(vertex)
    model.base.add(FormulaVertex(None, "c", 2))
    model.update_index()
    return model

if __name__ == "__main__":
    # Создание графа
    tree = get_lin_model(np.array([2.0, 1.0]))
    print(tree.to_strw())
    print(str(tree))
    print(tree.to_str_oper())
    print(tree.to_str_connect())
    print("Complexity = {}".format(tree.complexity()))

    # Расчёт графа
    inp = np.array([1, 2, 3])
    print("Input:", inp)
    inp.shape = (inp.shape[0], 1)
    print("Predict:", tree.predict(inp))

    # Обучение
    print(); print("Pre fit:")
    targ = np.array([3, 5, 7])
    tree = get_lin_model(np.zeros(2))
    tree.set_data(inp, targ)
    print(" weight = {}".format(tree.init_weight()))
    print(" RMSE = {}".format(tree.target_rmse()))
    #print("weights={}".format(tree.weights))
    tree.fit(inp, targ, 5)

    print(); print("Post fit:")
    print(" RMSE = {}".format(tree.target_rmse()))
    print(" weights = {}".format(tree.weights))
    print(" Formula:", str(tree))
    print(" Predict:", tree.predict(inp))

    # Сокращение графа
    print(); print("Reduction:")
    tree = get_lin_model2()
    print("Было: ", tree.to_strw())
    tree.reduction([1, 2, 4])
    print("Стало: ", tree.to_strw())
    print(str(tree))
    print(tree.to_str_oper())
    print(tree.to_str_connect())
    print("Complexity = {}".format(tree.complexity()))
