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
from .FormulaVertex import FormulaVertex

class FormulaTree:
    """
    Дерево операций, состоящие из вершин FormulaVertex.
    Args:
        base - (FormulaVertex) - Корень дерева.
        weights - (ndarray) - Вектор весов.
        weights_num - (int) - Если вектор весов не задан, инициализирует нулевой вектор весов.
        input_num - (int) - Количество входов.
        constants - (list) - Список констант. Если не задан, по умолчанию берутся константы CONSTANTS_OPERATIONS.
    """
    def __init__(self, base=None, weights=None, weights_num=1, input_num=1, constants=None):
        self.base = base
        self.weights = weights if type(weights) == np.ndarray else np.zeros(weights_num, dtype=float)
        self.constants = constants if isinstance(constants, list) else CONSTANTS_OPERATIONS
        self.input_num = input_num
        self.input = np.zeros(input_num, dtype=float)
        self.targ = np.zeros(1, dtype=float)
        self.ftarget = 0.0
        self.ftarget_min = 0.0
        self.complexity_target = 0.0
        self.ttarget = 0.0
        self.oper = np.array([]) #Список операций дерева
        self.connect1 = np.array([]) #Список связей (входящие связи)
        self.connect2 = np.array([]) #Список связей (исходящая связь)

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        not_copy = ["oper", "connect1", "connect2"]
        self.oper = np.array([])
        self.connect1 = np.array([])
        self.connect2 = np.array([])
        #print(" Deepcopy FormulaTree:")
        for key, val in self.__dict__.items():
            if not key in not_copy:
                setattr(result, key, copy.deepcopy(val, memo))
        self.update_index()
        return result

    # Вернуть случайный вес
    @staticmethod
    def get_rand_weight():
        return random.uniform(-2, 2)

    # Задать данные для расчётов
    def set_data(self, input=None, targ=None):
        self.input = input
        self.targ = targ

    # Инициализация дерева одним общим весом
    def init_weight(self, weight=None):
        if weight == None:
            weight = self.get_rand_weight()
        for ind in range(self.weights.shape[0]):
            self.weights[ind] = weight
        return weight

    # Инициализация весов дерева
    def init_weights(self):
        for ind in range(self.weights.shape[0]):
            self.weights[ind] = self.get_rand_weight()
        return self.weights

    # Расчёт дерева операций
    def predict(self, input):
        pred = np.zeros(input.shape[0], dtype=float)
        for ind in range(input.shape[0]):
            pred[ind] = self.base.calc(self.weights, input[ind], self.constants)
        pred[np.isnan(pred)] = 0
        return pred

    # Целевая функция
    def targetfun(self):
        self.ftarget = ((self.predict(self.input) - self.targ) ** 2).mean()
        return self.ftarget

    # Целевая функция для оптимизации
    def targetfun_opt(self, weights_opt):
        self.weights = weights_opt
        return self.targetfun()

    # Метрика MSE
    def target_mse(self):
        return ((self.predict(self.input) - self.targ) ** 2).sum() / self.targ.shape[0]

    # Метрика RMSE
    def target_rmse(self):
        return math.sqrt(self.target_mse())

    # Вычисление сложности дерева операций
    def complexity(self):
        self.complexity_target = self.base.complexity(0.0)
        return self.complexity_target

    # Целевая функция, учитывающая сложность
    def total_target(self, lmd=0.2):
        #self.ttarget = (1 - lmd) * self.targetfun() + lmd * self.complexity()
        self.ttarget = (1 - lmd) * self.ftarget + lmd * self.complexity_target
        return self.ttarget

    # Обучение весов дерева
    def fit(self, input, targ, maxiter, method='powell'):
        self.input = input
        self.targ = targ
        out = opt.minimize(self.targetfun_opt, self.weights[:], method=method, options={"maxiter": maxiter})
        self.weights = out.x.flatten()
        return out.x

    # Обновление индекса операций
    def update_oper(self):
        self.oper = np.array([])
        self.oper = self.base.update_oper(self.oper)

    # Обновление индекса связей
    def update_connect(self):
        self.connect1 = np.array([])
        self.connect2 = np.array([])
        self.connect1 = np.append(self.connect1, None)
        self.connect2 = np.append(self.connect2, self.base)
        (self.connect1, self.connect2) = self.base.update_connect(self.connect1, self.connect2)

    # Обновление всех индексов
    def update_index(self):
        self.update_oper()
        self.update_connect()

    # Очистить индекс
    def clear_index(self):
        self.oper = np.array([])
        self.connect1 = np.array([])
        self.connect2 = np.array([])

    # Проверка индексов
    def check_index(self):
        for ind_vert2 in range(self.oper.shape[0]-1):
            #print("oper", id(self.oper[ind_vert2]))
            vert2 = self.oper[ind_vert2]; #print("", "", ind_vert2)
            arr = np.where(self.connect2 == vert2); #print(arr[0])
            if len(arr[0]) != 1:
                print("check_index:", len(arr[0]))
            for ind in arr[0]:
                #print(id(self.connect1[ind]), id(self.connect2[ind]))
                vert1 = self.connect1[ind]
                jnd_rel = vert1.vertex_index(vert2)

    # Массив идентификаторов
    def get_ids(self):
        arr = []
        return self.base.get_ids(arr)

    # Сокращение операций
    def reduction(self, supp):
        self.base = self.base.reduction_req(None, supp)
        self.update_index()

    # Вывод индексов операций
    def to_str_oper(self):
        return [el.str_vertex() for el in self.oper]

    # Вывод индексов соединений
    def to_str_connect(self):
        arr_conn = []
        for ind in range(self.connect1.shape[0]):
            str1, str2 = ("None", "None")
            if self.connect1[ind] != None:
                str1 = self.connect1[ind].str_vertex()
            if self.connect2[ind] != None:
                str2 = self.connect2[ind].str_vertex()
            arr_conn.append("{} <- {}".format(str1, str2))
        return arr_conn

    # Преобразование дерева в строку
    def to_strw(self):
        st = str(self.base)
        for ind in range(len(self.constants)):
            st = st.replace("C[{}]".format(ind), str(self.constants[ind]))
        return st

    # Преобразование дерева в строку с заменой весов на числа
    def to_str(self):
        st = self.to_strw()
        for ind in range(self.weights.shape[0]):
            st = st.replace("W[{}]".format(ind), str(self.weights[ind]))
        return st

    # Преобразование дерева в строку с заменой весов на числа
    def __str__(self):
        return self.to_str()
