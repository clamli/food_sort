import numpy as np
import myutil as mtl
import math
import numpy as np
from copy import deepcopy


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, value_bound=10, data='random', pop_type='default', theta1=10.0, theta2=1.0, theta3=1.0, theta4=3.0):
        self.DNA_size = DNA_size          # food sequence size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size          # sequence pool size
        self.value_bound = value_bound    # health/taste bound
        #### define how to generate data
        if data == 'random':
            self.target = [np.random.randint(0, self.value_bound, size=2) for _ in range(self.DNA_size)]
            self.generate_init_data(pop_type, self.target)
        #### set necessary parameters for generic algorithm      
        self.left_bound = int(self.DNA_size/2)
        self.right_bound = self.left_bound
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4

    def set_data(self, pop_type, target):
        self.DNA_size = len(target)
        gen_result = self.generate_init_data(pop_type, target)
        self.left_bound = int(self.DNA_size/2)
        self.right_bound = self.left_bound
        return gen_result
        
    def change_parameters(self, theta1, theta2, theta3, theta4):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4

    def generate_init_data(self, pop_type, init_items):
        '''Generate different kinds of init data'''
        if pop_type == 'default':
            self.pop = np.vstack(np.random.permutation(init_items) for _ in range(self.pop_size)).reshape(self.pop_size, self.DNA_size, 2)
            return init_items
        ret_result = []
        taste_lst = [init_items[i][0] for i in range(len(init_items))]
        health_lst = [init_items[i][1] for i in range(len(init_items))]
        if pop_type == 'inverse proportional function':
            taste_lst.sort(reverse=True)
            health_lst.sort()
        elif pop_type == 'directly proportional function':
            taste_lst.sort()
            health_lst.sort()
        for t, h in zip(taste_lst, health_lst):
            ret_result.append([t, h])   
        self.pop = np.vstack(np.random.permutation(ret_result) for _ in range(self.pop_size)).reshape(self.pop_size, self.DNA_size, 2)
        return ret_result
        
    def get_fitness(self):
        fitness = np.empty((self.pop_size,), dtype=np.float64)
        leftarr = [self.pop[ind][0:self.left_bound+1].tolist() for ind in range(self.pop_size)]
        rightarr = [self.pop[ind][self.right_bound:].tolist() for ind in range(self.pop_size)]
        cost = np.empty((self.pop_size,), dtype=np.float64)
        middle_weigh = np.empty((self.pop_size,), dtype=np.float64)
        for leftlst, rightlst, popele, cnt in zip(leftarr, rightarr, self.pop, range(self.pop_size)):
            ###################### inverse pairing ######################
            left_taste_list, left_health_list = [leftlst[i][0] for i in range(len(leftlst))], [leftlst[i][1] for i in range(len(leftlst))]
            right_taste_list, right_health_list = [rightlst[i][0] for i in range(len(rightlst))], [rightlst[i][1] for i in range(len(rightlst))]
            # much more important
            left_taste_revcnt = mtl.CountInversions(left_taste_list, "decrease")
            right_health_revcnt = mtl.CountInversions(right_health_list, "increase")
            # less important
            left_health_revcnt = mtl.CountInversions(left_health_list, "increase")
            right_taste_revcnt = mtl.CountInversions(right_taste_list, "decrease")
            #################### neighbor difference ####################
            neighbor_difference = mtl.CalDifference(popele)
            ####################### middle weigh ########################
            middle_weigh[cnt] = popele[self.left_bound][0] + popele[self.left_bound][1]

            cost[cnt] = (self.theta1*(left_taste_revcnt+right_health_revcnt) \
                        + self.theta2*(left_health_revcnt+right_taste_revcnt) \
                            + self.theta3*neighbor_difference) 

        #### get fitness of all DNA
        fitness = pow(self.DNA_size*self.pop_size /(self.theta1+cost), 2) + self.theta4*middle_weigh
        return fitness

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            '''
            keep_city:
            array([[1, 2],
                   [0, 0],
                   [9, 1],
                   [6, 0]])
            '''
            keep_gene = parent[~cross_points]
            # print("----------", pop)
            # print("------", pop[i_][0])   
            # print("---", keep_city)                                    # find the city number
            swap_gene = pop[i_, mtl.isin(pop[i_][0].tolist(), keep_gene.tolist(), invert=True)]
            # print("parent", parent)
            # print("keep_gene", keep_gene)
            # print("swap_gene", swap_gene)
            parent[:] = np.concatenate((keep_gene, swap_gene))
        return parent

    def mutate(self, child):
        '''change position of any element in one sequence'''
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point].copy(), child[swap_point].copy()
                child[point], child[swap_point] = swapB, swapA
        return child
        # print("child", child[0])

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            # print("child", type(child[0]))
            parent[:] = child
        self.pop = pop
