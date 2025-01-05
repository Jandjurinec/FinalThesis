import random
import math
import numpy as np

import CNN

random.seed(42)

TRAINABLE_PARAMS = 4
LIMITS = [[0.001, 0.3],
          [3, 20], # int
          [0, 0.3],
          [0, 1]]

PRECISION = 4

ACC_IND = 4
F1_IND = 5

CNN = CNN.CNN_optimization()

def init_population(num_of_moths: int)-> np.ndarray:
    # lr, maxep, l2, mom, acc, f1
    m_position = np.zeros((num_of_moths, TRAINABLE_PARAMS + 2))
    for i in range(TRAINABLE_PARAMS):
       m_position[:, i] = np.round(np.random.uniform(0, 1, num_of_moths) * (LIMITS[i][0] - LIMITS[i][1]) + LIMITS[i][1], PRECISION)
    m_position[:, 1] = m_position[:, 1].astype(int)
    return m_position

def evaluate_moths(M: np.ndarray) -> None:
    for moth in M:
        moth[ACC_IND], moth[F1_IND] = CNN.get_metrics(moth[:TRAINABLE_PARAMS])

def stagnation_check(cc: np.ndarray, latest_t: int,  stagnation_similarity = 1e-4, last_n=4) -> bool:
    # if last_n values from stagnation curve are too similar, algorithm is stuck
    last_values = cc[latest_t - last_n:latest_t]
    
    # Check the differences between consecutive values
    for i in range(1, len(last_values)):
        if abs(last_values[i] - last_values[i-1]) > stagnation_similarity:
            return False
        
    return True


def MFO(num_of_moths = 7, MaxIter = 15, max_num_flames = 3):
    '''
        Moth flame optimizer algorithm
    '''
    M = init_population(num_of_moths)
    # M = import_latest_best_params(num_of_moths)
    Convergence_curve = np.zeros(MaxIter)
    best_flames = np.copy(M)
    double_M = np.zeros((2 * num_of_moths, TRAINABLE_PARAMS + 2))
    previous_M = np.zeros((num_of_moths, TRAINABLE_PARAMS + 2))
    t: int = 1

    while t < MaxIter + 1:
        flame_num = round(max_num_flames - t * ((max_num_flames - 1) / MaxIter))

        for i in range(0, num_of_moths):
            # Check if moths went out of the search space
            for j in range(TRAINABLE_PARAMS):
                M[i, j] = np.clip(M[i, j], LIMITS[j][0], LIMITS[j][1])
            M[:, 1] = M[:, 1].astype(int)

        # evaluate moths, NOTE: slow method
        evaluate_moths(M)
    
        if t == 1:
            # sort the array based on F1
            sorted_indices = np.argsort(M[:, F1_IND])[::-1]
            best_flames = M[sorted_indices, :]

        else:
            double_M = np.concatenate((previous_M, best_flames), axis=0)
            dobule_M_indices = np.argsort(double_M[:, F1_IND])[::-1]
            double_M_sorted = double_M[dobule_M_indices, :]

            best_flames = double_M_sorted[0:num_of_moths, :]
        
        # best flame obtained so far
        best_flame = best_flames[0]
        Convergence_curve[t-1] = best_flame[F1_IND]

        if t % 6 == 0:
            if stagnation_check(Convergence_curve, t, last_n=6):
                print("Algorithm stagnation")
                break

        previous_M = M

        a = -1 + t * ((-1) / MaxIter)
        
        for i in range(num_of_moths):
            for j in range(TRAINABLE_PARAMS):
                distance = abs(best_flames[i, j]- M[i,j])
                b = 1
                tt = (a-1) * random.random() + 1

                # focus bad moths into the flame
                if i <= flame_num:
                    M[i, j] = round(distance * math.exp(b * tt) * math.cos(tt * 2 * math.pi) + best_flames[i, j], PRECISION)
                else:
                    M[i, j] = round(distance * math.exp(b * tt) * math.cos(tt * 2 * math.pi) + best_flames[flame_num, j], PRECISION)

        if t % 1 == 0:
            print(f'Iteration: {t:02d} | best_f1: {best_flame[F1_IND]} | {best_flame}')
        
        t += 1

    print(Convergence_curve)
    return best_flame, Convergence_curve

## problems: initial population, algo can be stuck in local optima
