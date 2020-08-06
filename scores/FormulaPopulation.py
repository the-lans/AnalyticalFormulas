"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import random
import copy
import time
import scipy.optimize as opt

from .func import *
from .nsga_sort import nsga_sort
from .FormulaTree import FormulaTree
from .FormulaTreePopul import FormulaTreePopul

class FormulaPopulation:
    """
    Популяция особей: оптимизация весов, генетический алгоритм, обучение и предсказание.
    Args:
        input_num - (int) - Количество входов.
        weights_num - (int) - Задаёт вектор весов.
        constants - (list) - Список констант. Если не задан, по умолчанию берутся константы CONSTANTS_OPERATIONS.
        my_func - (list) - Список допустимых функций. Если не задан, по умолчанию берутся функции FUNC_OPERATIONS.
        my_prob - (list) - Задаёт важность использования соответствующих функций (в условных единицах). Если не задан,
            по умолчанию важность одинакова (1/len(my_func)).
        weights_popul - (int) - Количество испытаний, проведённых с одной особью.
        prob_func - (float) - Вероятность изменения операции дерева в особи.
        prob_vertex - (float) - Вероятность добавления новой вершины в особь.
        prob_crossover - (float) - Вероятность применения кроссовера к особи.
        cull_ratio - (float) - Отбор - устранение худших особей из племенного пула.
        elite_ratio - (float) - Элитарность - сохранить лучших особей без изменений.
        alg_probMoo - (float) - Методика ранжирования по Парето: mean-complexity или mean-min.
        prob_reduction - (float) - Вероятность сокращения операций в особи.
        lmd - (float) - Коэффициент общей целевой функции.
    """
    def __init__(self, input_num, weights_num, constants=None, my_func=None, my_prob=None, weights_popul=100, prob_func=0.2,
                 prob_vertex=0.2, prob_crossover=0.6, cull_ratio=0.1, elite_ratio=0.1, alg_probMoo=0.8, prob_reduction=0.0, lmd=0.2):
        self.my_func = my_func if isinstance(my_func, list) else FUNC_OPERATIONS
        self.my_args = get_my_args(my_func) if isinstance(my_func, list) else ARGS_OPERATIONS
        self.my_prob = weighted_prob(my_prob, len(my_func))
        #self.popul_num = popul
        self.input_num = input_num
        self.weights_num = weights_num
        self.constants = constants if isinstance(constants, list) else CONSTANTS_OPERATIONS
        self.population = []
        #self.start_popul(popul)
        self.weights_popul = weights_popul #Количество испытаний, проведённых с одной особью
        self.prob_func = prob_func #Изменить операцию дерева
        self.prob_vertex = prob_vertex #Добавить новую вершину
        self.prob_crossover = prob_crossover #Применить кроссовер
        self.cull_ratio = cull_ratio #Отбор - устранение худших особей из племенного пула
        self.elite_ratio = elite_ratio #Элитарность - сохранить лучших особей без изменений
        self.alg_probMoo = alg_probMoo #Методика ранжирования по Парето
        self.prob_reduction = prob_reduction #Сокращение операций
        self.select_tournSize = 2 #Особи для турнирного метода
        self.lmd = lmd #Коэффициент общей целевой функции
        self.vecsupport = [] #Опорный вектор значений
        self.support()
        self.is_print_iter = False #Вывод дополнительной информации в процессе обучения

    # Инициализация популяции
    def init_popul(self, popul):
        self.start_popul(popul)
        return self

    # Создание стартовой популяции
    def start_popul(self, popul):
        self.population = [FormulaTreePopul(self.my_func, self.my_args, self.my_prob, self.input_num, self.weights_num, self.constants).init_popul()
                           for _ in range(popul)]

    # Создание стартовой популяции по функции
    def start_popul_func(self, popul, func):
        self.population = []
        for _ in range(popul):
            tree = FormulaTreePopul(self.my_func, self.my_args, self.my_prob, self.input_num, self.weights_num, self.constants)
            tree = func(tree)
            tree.update_index()
            self.population.append(tree)

    # Проверка индексов
    def check_index(self, children=None):
        if not isinstance(children, list):
            children = self.population
        for ind in range(len(children)):
            #print("", ind)
            children[ind].check_index()

    # Кроссовер
    @staticmethod
    def crossover(parentA, parentB):
        """
        Объединить гены двух особей, чтобы произвести новую
        """
        childA = copy.deepcopy(parentA)
        if parentA.oper.shape[0] > 1 and parentB.oper.shape[0] > 1:
            childB = copy.deepcopy(parentB)
            objA = random.choice(childA.oper[:-1])
            objB = random.choice(childB.oper[:-1])
            arrA = np.where(childA.connect2 == objA)
            #arrB = np.where(childB.connect2 == objB)
            for ind in arrA[0]:
                vert = childA.connect1[ind]
                jnd_rel = vert.vertex_index(objA)
                vert.relations[jnd_rel] = objB
            childA.update_index()
        return childA

    # Оценка популяции по разным весам
    def multi_targetfun(self, input, targ):
        total_w = [FormulaTree.get_rand_weight() for _ in range(self.weights_popul)]; #print(total_w)
        for tree in self.population:
            tree.set_data(input, targ)
            ftarget = np.array([])
            ttarget = np.array([])
            for weight in total_w:
                tree.init_weight(weight)
                #tree.predict(input)
                ftarget = np.append(ftarget, tree.targetfun())
                ttarget = np.append(ttarget, tree.total_target(self.lmd))
            tree.complexity_target = tree.complexity()
            tree.ftarget = ftarget.mean(); #print(ftarget)
            tree.ftarget_min = ftarget.min()
            tree.ttarget = ttarget.mean()

    # Обучение популяции
    def targetfit(self, input, targ, maxiter, method='powell'):
        for tree in self.population:
            tree.init_weights()
            tree.fit(input, targ, maxiter, method)
            tree.complexity_target = tree.complexity()
            #tree.ftarget = tree.ftarget
            tree.ftarget_min = tree.ftarget
            tree.ttarget = tree.ftarget

    # Ранжирование
    def probMoo(self):
        """
        Ранжирование популяции по доминированию Парето.
        """
        # Compile objectives
        meanFit = np.asarray([ind.ftarget for ind in self.population])
        minFit = np.asarray([ind.ftarget_min for ind in self.population])
        nConns = np.asarray([ind.complexity_target for ind in self.population])
        nConns[nConns < 1] = 1  # No conns is always pareto optimal (but boring)
        eps = np.finfo(np.float32).eps
        objVals = np.c_[1/(meanFit + eps), 1/(minFit + eps), 1/nConns]  # Maximize
        #objVals = np.c_[-meanFit, -minFit, 1/nConns]  # Maximize

        # Alternate second objective
        if is_prob(self.alg_probMoo):
            rank = nsga_sort(objVals[:, [0, 2]])
            type_fit = "mean-complexity"
        else:
            rank = nsga_sort(objVals[:, [0, 1]])
            type_fit = "mean-min"

        # Assign ranks
        for ind in range(len(self.population)):
            self.population[ind].rank = rank[ind]
        return type_fit

    # Сортировка всех особей по рангу
    def rank_sort(self):
        self.population.sort(key=lambda x: x.rank)

    # Выбор особей турнирным метогдом
    def tournament(self, nOffspring):
        parentA = np.random.randint(len(self.population), size=(nOffspring, self.select_tournSize))
        parentB = np.random.randint(len(self.population), size=(nOffspring, self.select_tournSize))
        parents = np.vstack((np.min(parentA, 1), np.min(parentB, 1)))
        parents = np.sort(parents, axis=0)  # Higher fitness parent first
        return parents

    # Исключение низших особей и передача высших без изменений
    def cull_elite(self, ret_num=False):
        nOffspring = len(self.population)
        children = []

        # Cull  - eliminate worst individuals from breeding pool
        numberToCull = int(np.floor(self.cull_ratio * len(self.population)))
        if numberToCull > 0:
            self.population[-numberToCull:] = []

        # Elitism - keep best individuals unchanged
        nElites = int(np.floor(self.elite_ratio * len(self.population)))
        for ind in range(nElites):
            children.append(self.population[ind])
            nOffspring -= 1

        if ret_num:
            return (children, nOffspring, numberToCull, nElites)
        else:
            return (children, nOffspring)

    # Следующее поколение особей
    def recombine(self):
        """Создаёт следующее поколение особей
        Процедура:
        1) Сортировка всех особей по рангу
        2) Исключить более низкий процент особей из племенного пула
        3) Передать верхний процент особей в дочернюю популяцию без изменений
        4) Выбор особей турнирным методом
        5) Создание новой популяции через кроссовер и мутацию
        """
        # Ранжирование
        self.rank_sort()

        # Первоначальный отбор особей
        children, nOffspring = self.cull_elite()
        begChild = len(children)

        # Выбор особей турнирным методом
        parents = self.tournament(nOffspring)

        # Следующее поколение
        for ind in range(nOffspring):
            # Кроссовер
            #print(" "+str(ind))
            if is_prob(self.prob_crossover):
                child = self.crossover(self.population[parents[0, ind]], self.population[parents[1, ind]])
            else:
                child = copy.deepcopy(self.population[parents[0, ind]])
            children.append(child)
        #print("crossover", self.check_set()); self.check_index(children)

        # Изменение случайной операции дерева
        if not math.isclose(self.prob_func, 0.0):
            for ind in range(begChild, begChild+nOffspring):
                if is_prob(self.prob_func):
                    children[ind].rand_change_oper()
        #print("rand_change_oper", self.check_set()); self.check_index(children)

        # Добавление новой вершины в дерево
        if not math.isclose(self.prob_vertex, 0.0):
            for ind in range(begChild, begChild+nOffspring):
                if is_prob(self.prob_vertex):
                    #print("#{}".format(ind+1))
                    children[ind].rand_add_vertex()
        #print("add_vertex", self.check_set()); self.check_index(children)

        # Сокращение операций
        if not math.isclose(self.prob_reduction, 0.0):
            for ind in range(begChild, begChild + nOffspring):
                if is_prob(self.prob_reduction):
                    children[ind].reduction(self.vecsupport)

        return children

    # Сокращение операций
    def reduction(self):
        for tree in self.population:
            tree.reduction(self.vecsupport)

    # Этап эволюции особей
    def ask(self, input, targ):
        self.probMoo()
        self.population = self.recombine()
        self.multi_targetfun(input, targ)

    # Этап эволюции особей
    def askfit(self, input, targ, maxiter, method='powell'):
        self.probMoo()
        self.population = self.recombine()
        self.targetfit(input, targ, maxiter, method)

    # Последнее ранжирование популяции
    def sort(self):
        #self.multi_targetfun(input, targ)
        self.probMoo()
        self.rank_sort()

    # Запуск оптимизации
    def run(self, input, targ, iter, iterfit=0, method='powell', is_reduction=True):
        self.multi_targetfun(input, targ)
        for ind in range(iter):
            start_time = time.process_time()
            self.ask(input, targ)
            self.print_iter(ind, self.population[0], start_time)
        if iterfit > 0:
            self.runfit(input, targ, iterfit, iter=1, method=method)
        if is_reduction:
            self.reduction()
        self.sort()

    # Запуск оптимизации
    def runfit(self, input, targ, iterfit, iter=1, method='powell', is_reduction=True):
        self.targetfit(input, targ, iterfit, method)
        for ind in range(iter):
            start_time = time.process_time()
            self.askfit(input, targ, iterfit, method)
            self.print_iter(ind, self.population[0], start_time)
        if is_reduction:
            self.reduction()
        self.sort()

    # Обучение весов дерева
    def fit(self, input, targ, maxiter, method='powell'):
        for tree in self.population:
            tree.fit(input, targ, maxiter, method)

    # Предсказание
    def predict(self, input):
        result = np.array([])
        for tree in self.population:
            result = np.append(result, tree.predict(input))
        return result.reshape((len(self.population), input.shape[0]))

    # Нахождение опорных значений
    def support(self):
        self.vecsupport = []
        self.vecsupport = add_index(self.constants, self.vecsupport, 0) #0
        self.vecsupport = add_index(self.constants, self.vecsupport, 1) #1
        self.vecsupport = add_index(self.constants, self.vecsupport, 2) #2
        self.vecsupport = add_index(self.constants, self.vecsupport, -1) #3

    # Все веса популяции
    def get_weights(self):
        result = np.array([])
        for tree in self.population:
            result = np.append(result, tree.weights)
        return result.reshape((len(self.population), self.weights_num))

    # Информация про выполнение итерации
    def print_iter(self, ind, pl, start_time):
        if self.is_print_iter:
            str_format = "{:d} Formula={} Target={} TargetMin={} Complexity={} Time={:g}"
            print(str_format.format(ind+1, pl.to_strw(), pl.ftarget, pl.ftarget_min, pl.complexity_target, time.process_time()-start_time))

    # Проверка пересечений
    def check_set(self):
        for ind1 in range(len(self.population)):
            for ind2 in range(ind1+1, len(self.population)):
                st = set(self.population[ind1].get_ids()) & set(self.population[ind2].get_ids())
                if st != set():
                    return (ind1, ind2, st)
        return (None, None, set())

    # Преобразование популяции в строку с заменой весов на числа
    def to_str(self):
        vstr = ""
        for tree in self.population:
            vstr += str(tree) + "\n"

    # Преобразование популяции в строку с заменой весов на числа
    def __str__(self):
        return self.to_str()
