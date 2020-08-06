"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import random
import copy
import scipy.optimize as opt

from .func import *
from .FormulaVertex import FormulaVertex
from .FormulaTree import FormulaTree

class FormulaTreePopul(FormulaTree):
    """
    Дерево-особь для использования в генетическом алгоритме. Наследует класс FormulaTree.
    Args:
        my_func - (list) - Список допустимых функций.
        my_args - (list) - Количество аргументов у соответствующих функций.
        my_prob - (list) - Задаёт важность использования соответствующих функций (в условных единицах).
        input_num - (int) - Количество входов.
        weights_num - (int) - Задаёт вектор весов.
        constants - (list) - Список констант. Если не задан, по умолчанию берутся константы CONSTANTS_OPERATIONS.
    """
    def __init__(self, my_func, my_args, my_prob, input_num, weights_num=1, constants=None):
        super().__init__(weights_num=weights_num, input_num=input_num, constants=constants)
        self.my_func = my_func
        self.my_args = my_args
        self.my_prob = weighted_prob(my_prob)
        #self.weights = np.zeros(weights_num, dtype=float)
        #self.input_num = input_num
        #self.input = np.zeros(input_num, dtype=float)
        #self.targ = np.zeros(1, dtype=float)
        #self.start_popul()
        #self.update_index()
        self.rank = 0

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        #print(); print(" Deepcopy FormulaTreePopul:")
        for key, val in self.__dict__.items():
            setattr(result, key, copy.deepcopy(val, memo))
        self.update_index()
        return result

    # Инициализация дерева
    def init_popul(self):
        self.start_popul()
        self.update_index()
        return self

    # Создание стартовой популяции
    def start_popul(self):
        ind = weighted_select(self.my_prob)
        self.base = FormulaVertex(self.my_func[ind], typeval="o", indexval=ind)
        if self.my_args[ind] == 1:
            x_indval = random.randint(0, self.input.shape[0] - 1)
            self.base.add(FormulaVertex(None, "x", x_indval))
        else:
            wjnd = random.randint(0, self.my_args[ind] - 1)
            for jnd in range(self.my_args[ind]):
                if jnd == wjnd:
                    w_indval = random.randint(0, self.weights.shape[0] - 1)
                    self.base.add(FormulaVertex(None, "w", w_indval))
                else:
                    x_indval = random.randint(0, self.input.shape[0] - 1)
                    self.base.add(FormulaVertex(None, "x", x_indval))

    # Случайная функция из предложенного массива
    def rand_func(self, choice_func=None, choice_prob=None, this_func=None):
        is_list = (type(choice_func) == np.ndarray)
        ind = weighted_select(choice_prob) if is_list else weighted_select(self.my_prob)
        new_func = choice_func[ind] if is_list else self.my_func[ind]
        while new_func == this_func:
            ind = weighted_select(choice_prob) if is_list else weighted_select(self.my_prob)
            new_func = choice_func[ind] if is_list else self.my_func[ind]
        return (new_func, list(self.my_func).index(new_func))

    # Изменение случайной операции дерева
    def rand_change_oper(self):
        obj = random.choice(self.oper)
        filter = (np.array(self.my_args) == obj.relation_num)
        choice_func = np.array(self.my_func)[filter]
        choice_prob = np.array(self.my_prob)[filter]
        (obj.func, obj.indexval) = self.rand_func(choice_func, weighted_prob(choice_prob), obj.func)

    # Создание новой связи
    def new_relation(self, ioper):
        (indarr, indexval) = get_rand_inds([self.input_num, self.weights.shape[0], len(self.constants), ioper])
        if indarr == 0:
            return FormulaVertex(typeval="x", indexval=indexval)
        elif indarr == 1:
            return FormulaVertex(typeval="w", indexval=indexval)
        elif indarr == 2:
            return FormulaVertex(typeval="c", indexval=indexval)
        elif indarr == 3:
            return copy.deepcopy(self.oper[indexval])
            #return self.oper[indexval]
        return None

    # Новая вершина
    def new_vertex(self):
        (new_func, indexval) = self.rand_func()
        new_vertex = FormulaVertex(new_func, typeval="o", indexval=indexval)
        new_args = self.my_args[self.my_func.index(new_func)]
        return (new_vertex, new_args)

    # Добавление новой вершины в дерево
    def add_vertex(self, obj1, obj2, pVertex):
        (new_vertex, new_args) = pVertex
        ioper = self.oper.shape[0] if obj1 == None else list(self.oper).index(obj1); #print("ioper =", ioper)
        self.oper = np.insert(self.oper, ioper, new_vertex)

        # Связи у новой вершины
        jnd_args = random.randint(0, new_args-1)
        for jnd in range(new_args):
            vert = obj2 if jnd == jnd_args else self.new_relation(ioper)
            new_vertex.add(vert)
            self.connect1 = np.append(self.connect1, new_vertex)
            self.connect2 = np.append(self.connect2, vert)

        # Корректировка старых связей
        iconn = list(self.connect1).index(obj1)
        if obj1 == None:
            self.base = new_vertex
        else:
            jnd_rel = obj1.vertex_index(obj2)
            obj1.relations[jnd_rel] = new_vertex
        #self.connect1[iconn] = obj1
        self.connect2[iconn] = new_vertex

    # Добавление случайной вершины
    def rand_add_vertex(self):
        ind = random.randint(0, self.connect1.shape[0] - 1)
        obj1, obj2 = (self.connect1[ind], self.connect2[ind])
        #print("None" if obj1 == None else id(obj1), "None" if obj2 == None else id(obj2), obj1, obj2)
        self.add_vertex(obj1, obj2, self.new_vertex())
        self.update_index()
