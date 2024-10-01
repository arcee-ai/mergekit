import pandas as pd
import numpy as np
import random

ROOT = ''

class DE():
    def __init__(
            self,
            fct,
            D_x,
            lb,
            ub,
            max_fevals=10000,
            seed=0,
            F=0.7, Cr=0.5,
            population_size=10,
            generations=20):
        """

        :param fct: objective function f(x)
        :param lb: lower bounds
        :param ub: upper bounds
        :param D_x: number of dimensions in X space
        :param max_fevals: maximum number of function evaluations
        :param seed: seed for random generator
        :param F: weighted factor 
        """

        self._num_fevals = 0

        self._best = []
        self._best_f = float("inf")  # minimax
        
        self._fct = fct
        self._lb = lb
        self._ub = ub
        self._F = F
        self._Cr = Cr
        self._D_x = D_x
        self._max_fevals = max_fevals
        self._seed = seed
        self._population_size = population_size
        self._generations = generations

        random.seed(self._seed)
    
    def _DE(self):

        bests = []
        pop_mean = []
        pop_median = []

        popx = []
        ###GERA ALEATORIAMENTE A POPULACAO###
        for i in range(self._population_size):
            popx.append([random.random() for _ in range(self._D_x)])
        
        # avalia pop inicial - fpop
        fpop = self._fct(np.array(popx, dtype=float, copy=True))

        for i in range(self._population_size):
            if fpop[i] < self._best_f:
                self._best_f = fpop[i]
                self._best = popx[i]

        t = self._population_size
        while t < self._max_fevals:

            bests.append(self._best_f)
            pop_mean.append(np.mean(popx))
            pop_median.append(np.median(popx))
            
            pop_mutation = np.zeros((self._population_size, self._D_x))
            
            for i in range(self._population_size):
                alpha, beta, gamma = random.sample(range(self._population_size), 3)
                
                #mutacao
                pop_mutation[i] = np.asarray(popx[gamma]) + self._F*(np.asarray(popx[alpha]) - np.asarray(popx[beta]))
                
                #for m in pop_mutation[i]:
                #    if m < 0: m = 0
                #    if m > 1: m = 1

                for j in range(self._D_x):
                    if pop_mutation[i][j] < self._lb: 
                        pop_mutation[i][j] = 0
                    if pop_mutation[i][j] > self._ub: 
                        pop_mutation[i][j] = 1
                
            #cruzamento
            for i in range(self._population_size):
                for j in range(self._D_x):
                    rand = random.random()
                    if rand > self._Cr:
                        pop_mutation[i][j] = popx[i][j]
            
            #selecao            
            fmutado = self._fct(np.array(pop_mutation, dtype=float, copy=True))
            t += self._population_size
            for i in range(self._population_size):
                mut_min = fmutado[i]
                
                if mut_min < fpop[i]:
                    fpop[i] = mut_min
                    popx[i] = pop_mutation[i]

                    if mut_min < self._best_f:
                        self._best_f = mut_min
                        self._best = pop_mutation[i]

        pd.DataFrame({'best': bests}).to_csv(f'{ROOT}/DE_bests_{self._seed}.csv', index=False)
        pd.DataFrame({'pop_mean': pop_mean}).to_csv(f'{ROOT}/DE_popmean_{self._seed}.csv', index=False)
        pd.DataFrame({'pop_median': pop_median}).to_csv(f'{ROOT}/DE_popmedian_{self._seed}.csv', index=False)

        return self._best, self._best_f
    
    def _get_probability_strategy(self, m_success, m_fail, G, lp, k=4):
        if G >= lp:
            id1 = G - lp

            sc = np.array(m_success[id1:]) + 1e-3 # lp + (G-lp) x k
            fl = np.array(m_fail[id1:]) + 1e-3 # lp + (G-lp) x k

            s = (np.sum(sc, axis=0))/\
                (np.sum(sc, axis=0) + np.sum(fl, axis=0))  # 1 x k

            p = s / (np.sum(s, axis=0)) # 1xk

            return p
        else:
            raise ValueError('Generation number must be greater then learning period')

    def _mutation_rand_1(self, pop, F):
        alpha, beta, gamma = random.sample(range(self._population_size), 3)

        new_individual = np.asarray(pop[gamma]) + \
                        F*(np.asarray(pop[alpha]) - np.asarray(pop[beta]))
        
        return new_individual
    
    def _mutation_rand_to_best_2(self, pop, F, id_individual):
        x = pop[id_individual]

        r1, r2, r3, r4 = random.sample(range(self._population_size), 4)

        new_individual = np.asarray(x) + \
                        F*(np.asarray(self._best) - np.asarray(x)) +\
                        F*(np.asarray(pop[r1]) - np.asarray(pop[r2])) +\
                        F*(np.asarray(pop[r3]) - np.asarray(pop[r4]))

        return new_individual

    def _mutation_rand_2(self, pop, F):

        r1, r2, r3, r4, r5 = random.sample(range(self._population_size), 5)

        new_individual = np.asarray(pop[r1]) + \
                        F*(np.asarray(pop[r2]) - np.asarray(pop[r3])) +\
                        F*(np.asarray(pop[r4]) - np.asarray(pop[r5]))

        return new_individual

    def _mutation_current_to_rand_1(self, pop, F, id_individual):

        x = pop[id_individual]

        r1, r2, r3 = random.sample(range(self._population_size), 3)

        new_individual = np.asarray(x) + \
                        F*(np.asarray(pop[r1]) - np.asarray(x)) +\
                        F*(np.asarray(pop[r2]) - np.asarray(pop[r3]))

        return new_individual

    def _mutation(self, strat, F, pop, id_individual):

        if strat == 1:
            return self._mutation_rand_1(pop, F)
        elif strat == 2:
            return self._mutation_rand_to_best_2(pop, F, id_individual)
        elif strat == 3:
            return self._mutation_rand_2(pop, F)
        elif strat == 4:
            return self._mutation_current_to_rand_1(pop, F, id_individual)
        else:
            raise ValueError('Id de estratégia de mutacao inválido')

    def _SaDE(self, learning_period):

        learning_period = (self._max_fevals / self._population_size)*0.2

        bests = []
        pop_mean = []
        pop_median = []
        
        CR = []
        CRm = []
        memory_cr = [[], [], [], []]
        prob_strategy = []
        strategy_success = []
        strategy_fail = []
        learning_period = 2
        F = []
        strategies = [
            'DE/rand/1/bin',
            'DE/rand-to-best/2/bin',
            'DE/rand/2/bin',
            'DE/current-to-rand/1'
        ]
        n_strategies = 4
        generation = 0

        popx = []
        ###GERA ALEATORIAMENTE A POPULACAO###
        for i in range(self._population_size):
            popx.append([random.random() for _ in range(self._D_x)])
        
        # avalia pop inicial - fpop
        fpop = self._fct(np.array(popx, dtype=float, copy=True))

        for i in range(self._population_size):
            if fpop[i] < self._best_f:
                self._best_f = fpop[i]
                self._best = popx[i]

        t = self._population_size
        while t < self._max_fevals:

            bests.append(self._best_f)
            pop_mean.append(np.mean(popx))
            pop_median.append(np.median(popx))

            chosen_strategies = []

            F.append([np.random.normal(0.5, 0.3) for _ in range(n_strategies)])
            
            # calculate prob
            if generation > learning_period:
                p = self._get_probability_strategy(
                    strategy_success,
                    strategy_fail,
                    generation,
                    learning_period
                )
                prob_strategy.append(p.tolist())

                CRm.append([np.median(memory_cr[k]) for k in range(n_strategies)])
            else:
                prob_strategy.append([1/n_strategies for _ in range(n_strategies)])

                CRm.append([0.5, 0.5, 0.5, 0.5])
            
            cr = []
            for k in range(n_strategies):
                aux = []
                for j in range(self._population_size):
                    r = np.random.normal(CRm[generation][k], 0.1)
                    while (r < 0) | (r > 1):
                        r = np.random.normal(CRm[generation][k], 0.1)
                    aux.append(r)
                cr.append(aux)
            CR.append(cr)

            pop_mutation = np.zeros((self._population_size, self._D_x))
            for i in range(self._population_size):
                # escolha estrategia
                strat = np.random.choice([i for i in range(1, n_strategies+1)], p=prob_strategy[generation])
                pop_mutation[i] = self._mutation(
                    strat, 
                    F[generation][strat-1],
                    popx,
                    i
                )
                chosen_strategies.append(strat)

                for j in range(self._D_x):
                    if pop_mutation[i][j] < self._lb: 
                        pop_mutation[i][j] = 0
                    if pop_mutation[i][j] > self._ub: 
                        pop_mutation[i][j] = 1
            
            #cruzamento
            for i in range(self._population_size):
                strat = chosen_strategies[i] - 1
                for j in range(self._D_x):
                    rand = random.random()
                    if rand > CR[generation][strat][i]:
                        pop_mutation[i][j] = popx[i][j]

            fmutado = self._fct(np.array(pop_mutation, dtype=float, copy=True))
            t += self._population_size
            sucess = np.zeros(n_strategies)
            fail = np.zeros(n_strategies)
            for i in range(self._population_size):
                strat = chosen_strategies[i] - 1
                mut_min = fmutado[i]
                
                if mut_min < fpop[i]:
                    fpop[i] = mut_min
                    popx[i] = pop_mutation[i]

                    sucess[strat] += 1
                    memory_cr[strat].append(CR[generation][strat][i])

                    if mut_min < self._best_f:
                        self._best_f = mut_min
                        self._best = pop_mutation[i]
                else:
                    fail[strat] += 1

            strategy_success.append(sucess)
            strategy_fail.append(fail)

            generation+=1

        pd.DataFrame({'best': bests}).to_csv(f'{ROOT}/SaDE_bests_{self._seed}.csv', index=False)
        pd.DataFrame({'pop_mean': pop_mean}).to_csv(f'{ROOT}/SaDE_popmean_{self._seed}.csv', index=False)
        pd.DataFrame({'pop_median': pop_median}).to_csv(f'{ROOT}/SaDE_popmedian_{self._seed}.csv', index=False)

        return self._best, self._best_f

    def run_DE(self):

        print('Executando Evolução Diferencial')
        print(f'Seed: {self._seed}')
        
        return self._DE()
    
    def run_SaDE(self, learning_period):
        print('Executando Self Adaptative Evolução Diferencial')
        print(f'Seed: {self._seed}')
        
        return self._SaDE(learning_period)
        
