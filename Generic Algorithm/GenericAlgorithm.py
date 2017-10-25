import matplotlib.pyplot as plt
import myutil as mtl
import math
import numpy as np
from copy import deepcopy

DNA_SIZE = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.01
POP_SIZE = 500
N_GENERATIONS = 100


class GA(object):
	def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, value_bound=10, data_type='random', pop_type='default', theta1=10.0, theta2=1.0, theta3=1.0, theta4=3.0):
		self.DNA_size = DNA_size          # food sequence size
		self.cross_rate = cross_rate
		self.mutate_rate = mutation_rate
		self.pop_size = pop_size          # sequence pool size
		self.value_bound = value_bound    # health/taste bound
		#### define how to generate data
		if data_type == 'random':
			self.target = [np.random.randint(0, self.value_bound, size=2) for _ in range(self.DNA_size)]
		elif data_type == 'file':
			col_name_lst, nutrient_dict = load_nutrient_data('100_foods_calorie_and_core_nutrients.xlsx')
			self.target = combine_data('Energy (kcal)', 'Sodium (mg)', col_name_lst, nutrient_dict, plot='False')
		self.generate_init_data(pop_type, self.target)
		#### set necessary parameters for generic algorithm
		self.left_bound = int(DNA_size/2)
		self.right_bound = self.left_bound
		self.theta1 = theta1
		self.theta2 = theta2
		self.theta3 = theta3
		self.theta4 = theta4

	def change_parameters(self, theta1, theta2, theta3, theta4):
		self.theta1 = theta1
		self.theta2 = theta2
		self.theta3 = theta3
		self.theta4 = theta4

	def generate_init_data(self, pop_type, init_items):
		'''Generate different kinds of init data'''
		if pop_type == 'default':
			self.pop = np.vstack(np.random.permutation(init_items) for _ in range(self.pop_size)).reshape(self.pop_size, self.DNA_size, 2)
			return
		ret_result = []
		taste_lst = [init_items[i][0] for i in range(len(init_items))]
		health_lst = [init_items[i][1] for i in range(len(init_items))]
		if pop_type == 'inverse proportional function':
			taste_lst.sort()
			health_lst.sort(reverse=True)
		elif pop_type == 'directly proportional function':
			taste_lst.sort()
			health_lst.sort()
		for t, h in zip(taste_lst, health_lst):
			ret_result.append([t, h])
		self.pop = np.vstack(np.random.permutation(ret_result) for _ in range(self.pop_size)).reshape(self.pop_size, self.DNA_size, 2)

	def get_fitness(self):
		fitness = np.empty((self.pop_size,), dtype=np.float64)
		leftarr = [self.pop[ind][0:ga.left_bound+1].tolist() for ind in range(ga.pop_size)]
		rightarr = [self.pop[ind][ga.right_bound:].tolist() for ind in range(ga.pop_size)]
		cost = np.empty((self.pop_size,), dtype=np.float64)
		middle_weigh = np.empty((self.pop_size,), dtype=np.float64)
		for leftlst, rightlst, popele, cnt in zip(leftarr, rightarr, self.pop, range(ga.pop_size)):
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
		fitness = pow(POP_SIZE*DNA_SIZE / cost, 2) + self.theta4*middle_weigh
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


ga = GA(DNA_size=DNA_SIZE, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, value_bound=10, data_type='random', pop_type='default', theta1=1.0, theta2=0.0)
best_fit_lst1 = []
best_fit_lst2 = []
best_fit_lst3 = []
best_seq1 = []
best_seq2 = []
best_seq3 = []
max_seq1 = []
max_seq2 = []
max_se13 = []
plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

#### for default target ####
max_fitness = 0
for generation in range(N_GENERATIONS):
	fitness = ga.get_fitness()
	best_idx = np.argmax(fitness)
	print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx])
	print('Best Sort:\n', ga.pop[best_idx])
	best_fit_lst1.append(fitness[best_idx])
	if fitness[best_idx] >= max_fitness:
		max_seq1 = deepcopy(ga.pop[best_idx])
	ga.evolve(fitness)

#### for inverse proportional function target ####
max_fitness = 0
ga.generate_init_data('inverse proportional function', ga.target)
for generation in range(N_GENERATIONS):
	fitness = ga.get_fitness()
	best_idx = np.argmax(fitness)
	print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx])
	print('Best Sort:\n', ga.pop[best_idx])
	best_fit_lst2.append(fitness[best_idx])
	if fitness[best_idx] >= max_fitness:
		max_seq2 = deepcopy(ga.pop[best_idx])
	ga.evolve(fitness) 

#### for directly proportional function target ####	
max_fitness = 0
ga.generate_init_data('directly proportional function', ga.target)
for generation in range(N_GENERATIONS):
	fitness = ga.get_fitness()
	best_idx = np.argmax(fitness)
	print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx])
	print('Best Sort:\n', ga.pop[best_idx])
	best_fit_lst3.append(fitness[best_idx])
	if fitness[best_idx] >= max_fitness:
		max_seq3 = deepcopy(ga.pop[best_idx])
	ga.evolve(fitness)  

#### plot result ####
plt.figure(1)
plt.title("Best taste and health trend for default dataset")
plt.xlabel("item number")
plt.ylabel("taste/health value")
plt.plot(range(1, len(max_seq1)+1), [max_seq1[i][0] for i in range(len(max_seq1))], 'r-', label="taste_d", marker='*')
plt.plot(range(1, len(max_seq1)+1), [max_seq1[i][1] for i in range(len(max_seq1))], 'r--', label="health_d", marker='*')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.figure(2)
plt.title("Best taste and health trend for inverse proportional function  dataset")
plt.xlabel("item number")
plt.ylabel("taste/health value")
plt.plot(range(1, len(max_seq2)+1), [max_seq2[i][0] for i in range(len(max_seq2))], 'g-', label="taste_iv", marker='*')
plt.plot(range(1, len(max_seq2)+1), [max_seq2[i][1] for i in range(len(max_seq2))], 'g--', label="health_iv", marker='*')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.figure(3)
plt.title("Best taste and health trend for directly proportional function dataset")
plt.xlabel("item number")
plt.ylabel("taste/health value")
plt.plot(range(1, len(max_seq3)+1), [max_seq3[i][0] for i in range(len(max_seq3))], 'b-', label="taste_dp", marker='*')
plt.plot(range(1, len(max_seq3)+1), [max_seq3[i][1] for i in range(len(max_seq3))], 'b--', label="health_dp", marker='*')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)

#### plot result ####
plt.figure(4)
plt.title("Change of Best Fitness for different dataset")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.sca(ax1)
plt.plot(best_fit_lst1)
plt.sca(ax2)
plt.plot(best_fit_lst2)
plt.sca(ax3)
plt.plot(best_fit_lst3)

plt.show()