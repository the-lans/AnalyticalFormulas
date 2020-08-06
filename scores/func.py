"""
Copyright © 2020. All rights reserved.
Author: Vyshinsky Ilya <ilyav87@gmail.com>
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import random
import math

"""
Доступные по умолчанию функции дерева операций.
"""

def fsum(args):
    return args.sum()

def fsub(args):
    return args[0] - args[1]

def fmul(args):
    return np.dot(args[0], args[1])

def fdev(args):
    if math.isclose(args[0], 0.0):
        return 0.0
    elif math.isclose(args[1], 0.0):
        return 1.0e+150 * (1.0 if args[0] > 0 else -1.0)
    else:
        return args[0] / args[1]

def fpower(args):
    if math.isclose(args[1], 0.0):
        result = 1.0
    elif args[0] <= 0.0 and args[1] > 1:
        result = args[0] ** int(args[1])
    elif args[0] <= 0.0 and args[1] < -1:
        result = args[0] ** int(args[1])
        if math.isclose(result, 0.0):
            result = 1.0e+150 * (1.0 if result > 0 else -1.0)
    elif args[0] <= 0.0 and abs(args[1]) <= 1:
        result = 0.0
    else:
        result = args[0] ** args[1]
    return result

def fsin(args):
    return np.sin(args[0])

def fcos(args):
    return np.cos(args[0])

def ftanh(args):
    return np.tanh(args[0])

def farctan(args):
    return np.arctan(args[0])

def fsinc(args):
    if math.isclose(args[0], 0.0):
        return 1.0
    else:
        return np.sin(args[0]) / args[0]

def fgauss(args):
    return np.exp(-args[0] * args[0])

#def f(args):
#    return f(args[0])

"""
Глобальные определения для функций и констант.
"""

FUNC_OPERATIONS = [fsum, fsub, fmul, fdev, fsin, fcos, ftanh, farctan, fsinc, fgauss]
ARGS_OPERATIONS = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
STR_OPERATIONS = ['+', '-', '*', '/', 'sin', 'cos', 'tanh', 'atg', 'sinc', 'gauss']
COMPL_OPERATIONS = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
TWO_OPERATIONS = [fsum, fsub, fmul, fdev]
THREE_OPERATIONS = []
CONSTANTS_OPERATIONS = [-1, 0, 1, 1.41421356237, 2, 2.7182818284, 3, 3.1415926535]


# Отбор количества операций по функциям
def get_my_args(my_func):
    my_args = []
    for el in my_func:
        my_args.append(ARGS_OPERATIONS[FUNC_OPERATIONS.index(el)])
    return my_args

# Возвращает случайный индекс массива с учётом весов
def weighted_select(probs):
    pval = random.random()
    acc = 0
    for ind in range(len(list(probs))):
        acc += probs[ind]
        if pval <= acc:
            return ind
    return None

# Нормализует вектор весов (сумма весов = 1)
def weighted_prob(my_prob=None, num=1):
    if isinstance(my_prob, list):
        sum_prob = sum(my_prob)
        return [el / sum_prob for el in my_prob]
    elif isinstance(my_prob, np.ndarray):
        sum_prob = my_prob.sum()
        return [el / sum_prob for el in my_prob]
    else:
        return [1 / num for _ in range(num)]

# Возникновение случайного события
def is_prob(prob):
    return prob >= random.random()

# Возвращает случайную принадлежность к одному из массивов
def get_rand_inds(arrs):
    num = arrs.sum() if isinstance(arrs, np.ndarray) else sum(arrs)
    ind = random.randint(0, num-1)
    ind2 = ind
    isum = 0
    for jnd in range(len(list(arrs))):
        isum += arrs[jnd]
        if ind < isum:
            return (jnd, ind2)
        ind2 -= arrs[jnd]
    return (jnd, ind2)

# Добавляем индекс значения
def add_index(constants, vec, val):
    if val in constants:
        vec.append(constants.index(val))
    else:
        vec.append(None)
    return vec
