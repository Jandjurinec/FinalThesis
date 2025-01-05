import random
import numpy as np
import matplotlib.pyplot as plt

import CNN

random.seed(42)

TRAINABLE_PARAMS = 4
LIMITS = [[0.001, 0.5],
          [3, 20], # int
          [0, 0.3],
          [0, 1]]

PRECISION = 4

ACC_IND = 4
F1_IND = 5

VELOCITY_BOUNDARY = [5, 5, 5, 5]

CNN = CNN.CNN_optimization()

def get_init_position()-> np.ndarray:
    # lr, maxep, l2, mom, acc, f1
    m_position = np.zeros(TRAINABLE_PARAMS + 2)
    for i in range(TRAINABLE_PARAMS):
       m_position[i] = np.round(np.random.uniform(0, 1) * (LIMITS[i][0] - LIMITS[i][1]) + LIMITS[i][1], PRECISION)
    m_position[1] = m_position[1].astype(int)
    return m_position


def stagnation_check(cc: np.ndarray, latest_t: int,  stagnation_similarity = 1e-4, last_n=4) -> bool:
    # if last_n values from stagnation curve are too similar, algorithm is stuck
    last_values = cc[latest_t - last_n:latest_t]
    
    # Check the differences between consecutive values
    for i in range(1, len(last_values)):
        if abs(last_values[i] - last_values[i-1]) > stagnation_similarity:
            return False
        
    return True


def init_population(swarm_size: int) -> list:
    M = []
    for i in range(swarm_size):
        M.append(Particle())
    return M


def evaluate_swarm(M: np.ndarray) -> None:
    for particle in M:
        particle.position[ACC_IND], particle.position[F1_IND] = CNN.get_metrics(particle.position[:TRAINABLE_PARAMS])


class Particle:
    def __init__(self):
        self.position = get_init_position()
        self.velocity = np.ones(shape=TRAINABLE_PARAMS) * np.random.uniform(0, 1, (TRAINABLE_PARAMS))
        self.personal_best = np.copy(self.position)

    def __str__(self):
        return f"{self.position} | velocity: {self.velocity} | personal_best: {self.personal_best[F1_IND]}"



def PSO(swarm_size, w, c1, c2, max_iter, is_global_neigh=False):
    '''
        Particle swarm optimizer algorithm
    '''
    M = init_population(swarm_size)
    Convergence_curve = np.zeros(max_iter)
    Convergence_curve_particle = np.zeros(max_iter)
    global_best = Particle()

    for t in range(1, max_iter + 1):
            
        evaluate_swarm(M)

        # personal best update
        for particle in M:
            if particle.personal_best[F1_IND] == 0 or particle.position[F1_IND] > particle.personal_best[F1_IND]:
                particle.personal_best = particle.position.copy()
                # global best update
                if particle.personal_best[F1_IND] > global_best.personal_best[F1_IND]:
                    global_best.personal_best = particle.personal_best.copy()

        # update velocity
        for i in range(swarm_size):
            r1 = np.random.uniform(0,1, TRAINABLE_PARAMS)
            r2 = np.random.uniform(0,1, TRAINABLE_PARAMS)

            c1r1coef = (M[i].personal_best[:TRAINABLE_PARAMS] - M[i].position[:TRAINABLE_PARAMS]) * (c1 * r1)
            c2r2coef = 0

            if is_global_neigh:
                c2r2coef = (global_best.personal_best[:TRAINABLE_PARAMS] - M[i].position[:TRAINABLE_PARAMS]) * (c2 * r2)
            else:
                left = None
                right = None

                if i == 0:
                    left = M[-1]
                    right = M[i+1]
                elif i == swarm_size - 1:
                    left = M[i-1]
                    right = M[0]
                else:
                    left = M[i-1]
                    right = M[i+1]

                if left.personal_best[F1_IND] > right.personal_best[F1_IND]:
                    c2r2coef = (left.personal_best[:TRAINABLE_PARAMS] - M[i].position[:TRAINABLE_PARAMS]) * (c2 * r2)
                else:
                    c2r2coef = (right.personal_best[:TRAINABLE_PARAMS] - M[i].position[:TRAINABLE_PARAMS]) * (c2 * r2)

            # update velocity
            new_velocity = M[i].velocity * w + c1r1coef + c2r2coef
            for j in range(len(VELOCITY_BOUNDARY)):
                new_velocity[j] = np.clip(new_velocity[j], -VELOCITY_BOUNDARY[j], VELOCITY_BOUNDARY[j])
            
            M[i].velocity = new_velocity
            M[i].position[:TRAINABLE_PARAMS] +=  new_velocity

        for i in range(0, swarm_size):
            # Check if particle went out of the search space
            for j in range(TRAINABLE_PARAMS):
                M[i].position[j] = np.clip(M[i].position[j], LIMITS[j][0], LIMITS[j][1])

            M[i].position[1] = round(M[i].position[1])
            
        global_best_f1 = global_best.personal_best[F1_IND]
        Convergence_curve[t-1] = global_best_f1
        Convergence_curve_particle[t-1] = M[0].position[F1_IND]


        if t % 1 == 0:
            print(f'Iteration: {t:02d} | best_f1: {global_best_f1} | first_particle_f1: {M[0].position[F1_IND]} | first_particle_best_f1: {M[0].personal_best[F1_IND]}')

        # stagnation check
        if t % 15 == 0:
            if stagnation_check(Convergence_curve, t, last_n=6):
                print("Algorithm stagnation")
                break

    return global_best_f1, Convergence_curve, Convergence_curve_particle
        