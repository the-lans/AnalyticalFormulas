"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import random
import copy
import math
import scipy.optimize as opt

from .func import *

class FormulaVertex:
    """
    Класс вершины дерева операций. Хранит информацию: о связях с другими вершинами,
    вычисляемой функции, индексе данных и типе вершины.
    Args:
        func - (function) - Вычисляемая функция.
        typeval - (Str) - Тип вершины:
            x - значение входного вектора
            w - значение веса
            с - константное значение
            o - выполнение операции.
        indexval - (int) - Индекс значения: либо входного вектора, либо вектора весов,
            либо вектора констант.
        rel1, rel2, rel3 - (FormulaVertex) - Связи с другими вершинами.
    """
    def __init__(self, func=None, typeval="x", indexval=0, rel1=None, rel2=None, rel3=None):
        self.relations = [] #Хранит все связи с другими вершинами
        if rel1 != None: self.relations.append(rel1)
        if rel2 != None: self.relations.append(rel2)
        if rel3 != None: self.relations.append(rel3)
        self.relation_num = 0 #Количество связей
        self.update_relation_num()
        self.func = func #Вычисляемая функция
        self.typeval = typeval if func == None else "o" #Тип вершины
        self.indexval = indexval #Индекс значения

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        #print(" Deepcopy FormulaVertex:")
        for key, val in self.__dict__.items():
            if key == "relations":
                result.relations = [copy.deepcopy(el, memo) for el in val]; #print(key, result.relations)
            else:
                setattr(result, key, copy.deepcopy(val, memo)); #print(key, val)
        #print(" End Deepcopy FormulaVertex")
        return result

    def __del__(self):
        self.del_relations()

    # Удалить связи
    def del_relations(self):
        for rel in self.relations:
            del rel
        self.relations = []
        self.relation_num = 0

    # Вставить новую связь
    def add_relation(self, ind, rel):
        self.relations[ind] = rel
        self.update_relation_num()

    # Добавить связь в конец
    def add(self, rel=None):
        self.relations.append(rel)
        if rel != None:
            self.relation_num += 1

    # Найти смежную вершину
    def vertex_index(self, obj):
        return self.relations.index(obj)

    # Обновить используемое количество связей
    def update_relation_num(self):
        self.relation_num = 0
        for rel in self.relations:
            if rel == None:
                break
            else:
                self.relation_num += 1

    # Расчёт дерева операций
    def calc(self, weights, input, constants):
        if self.relation_num > 0:
            vals = np.zeros(self.relation_num, dtype=float)
            for ind in range(self.relation_num):
                vals[ind] = self.relations[ind].calc(weights, input, constants)
        if self.typeval == "x":
            return input[self.indexval]
        elif self.typeval == "w":
            return weights[self.indexval]
        elif self.typeval == "c":
            return constants[self.indexval]
        elif self.typeval == "o":
            return self.func(vals)

    # Вычисление сложности дерева операций
    def complexity(self, acc):
        for ind in range(self.relation_num):
            acc = self.relations[ind].complexity(acc)
        if self.typeval == "o":
            acc += COMPL_OPERATIONS[FUNC_OPERATIONS.index(self.func)]
        return acc

    # Обновление индекса операций
    def update_oper(self, oper):
        for ind in range(self.relation_num):
            oper = self.relations[ind].update_oper(oper)
        if self.typeval == "o" and not self in oper:
            oper = np.append(oper, self)
        return oper

    # Обновление индекса связей
    def update_connect(self, connect1, connect2):
        for ind in range(self.relation_num):
            connect1 = np.append(connect1, self)
            connect2 = np.append(connect2, self.relations[ind])
        for ind in range(self.relation_num):
            (connect1, connect2) = self.relations[ind].update_connect(connect1, connect2)
        return (connect1, connect2)

    # Вернуть ID
    def get_ids(self, arr):
        for rel in self.relations:
            arr = rel.get_ids(arr)
            arr.append(id(rel))
        arr.append(id(self))
        return arr

    # Рекурсивное сокращение операций
    def reduction_req(self, parent, supp):
        for ind in range(self.relation_num):
            self.relations[ind].reduction_req(self, supp)
        return self.reduction(parent, supp)

    # Сокращение операций
    def reduction(self, parent, supp):
        if self.typeval == "o" and (self.func in TWO_OPERATIONS):
            if self.eq_two():
                if self.func == fsum:
                    self.relations[0].set_const(supp[2])
                    self.set_func(fmul)
                elif self.func == fsub:
                    self.set_const(supp[0])
                elif self.func == fdev:
                    self.set_const(supp[1])
            elif self.eq_const(supp[0], 0) and self.func == fsum:
                return self.del_vertex(parent, 1)
            elif self.eq_const(supp[0], 1) and self.func == fsum:
                return self.del_vertex(parent, 0)
            elif self.eq_const(supp[0], 0) and self.func == fsub:
                self.relations[0].indexval = supp[3] #-1
                self.set_func(fmul)
            elif self.eq_const(supp[0], 1) and self.func == fsub:
                return self.del_vertex(parent, 0)
            elif self.eq_const(supp[1], 0) and self.func == fmul:
                return self.del_vertex(parent, 1)
            elif self.eq_const(supp[1], 1) and self.func == fmul:
                return self.del_vertex(parent, 0)
            elif self.eq_const(supp[1], 1) and self.func == fdev:
                return self.del_vertex(parent, 0)
        return self

    # Установить функцию
    def set_func(self, func, my_func=None):
        self.func = func
        self.indexval = my_func.index(func) if isinstance(my_func, list) else 0

    # Удалить текущую вершину
    def del_vertex(self, parent, ind_rel=None):
        if ind_rel == None:
            self.del_relations()
            return parent
        else:
            rel = self.relations[ind_rel]
            self.relations[ind_rel] = None
            self.del_relations()
            if parent == None:
                return rel
            else:
                ind = parent.vertex_index(self)
                parent.relations[ind] = rel
                return parent

    # Установить константу
    def set_const(self, indval):
        self.del_relations()
        self.typeval = "c"
        self.indexval = indval
        self.func = None

    # Проверка на одинаковые значения бинарных операций
    def eq_two(self):
        rel1, rel2 = (self.relations[0], self.relations[1])
        return (rel1.typeval != "o" and rel1.typeval == rel2.typeval and rel1.indexval == rel2.indexval)

    # Проверка значения константы
    def eq_const(self, val, indx=None):
        if indx == None:
            for ind in range(self.relation_num):
                rel = self.relations[ind]
                if rel.typeval == "c" and rel.indexval == val:
                    return True
        else:
            rel = self.relations[indx]
            if rel.typeval == "c" and rel.indexval == val:
                return True
        return False

    # Преобразование элемента в строку
    def __str__(self):
        if self.typeval == "o" and (self.func in TWO_OPERATIONS):
            return self.str_two()
        elif self.typeval == "o" and (self.func in THREE_OPERATIONS):
            return self.str_three()
        elif self.typeval == "o":
            return self.str_one()
        elif self.typeval == "x":
            return "X[{}]".format(self.indexval)
        elif self.typeval == "w":
            return "W[{}]".format(self.indexval)
        elif self.typeval == "c":
            return "C[{}]".format(self.indexval)
        return ""

    # Преобразование унарных (и префиксных) операций в строку
    def str_one(self):
        ind = FUNC_OPERATIONS.index(self.func)
        st = ""
        for jnd in range(self.relation_num):
            st += ", " + str(self.relations[jnd])
        return "{}({})".format(STR_OPERATIONS[ind], st[2:])

    # Преобразование бинарной операции в строку
    def str_two(self):
        ind = FUNC_OPERATIONS.index(self.func)
        return "({} {} {})".format(str(self.relations[0]), STR_OPERATIONS[ind], str(self.relations[1]))

    # Преобразование тернарной операции в строку
    def str_three(self):
        ind = FUNC_OPERATIONS.index(self.func)
        return "({} {} 0, {}, {})".format(str(self.relations[0]), STR_OPERATIONS[ind], str(self.relations[1]), str(self.relations[2]))

    # Вывод только текущей вершины
    def str_vertex(self):
        if self.typeval == "o":
            ind = FUNC_OPERATIONS.index(self.func)
            return "{}:{}:{}".format(id(self), STR_OPERATIONS[ind], ARGS_OPERATIONS[ind])
        elif self.typeval == "x":
            return "X[{}]".format(self.indexval)
        elif self.typeval == "w":
            return "W[{}]".format(self.indexval)
        elif self.typeval == "c":
            return "C[{}]".format(self.indexval)
        return ""
