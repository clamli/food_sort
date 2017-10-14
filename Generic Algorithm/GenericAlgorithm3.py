import matplotlib.pyplot as plt
import myutil as mtl
import math
import numpy as np

DNA_SIZE = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.01
POP_SIZE = 500
N_GENERATIONS = 1000


class GA(object):
	def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, value_bound=10, pop=None, theta1=10.0, theta2=1.0, theta3=1.0, theta4=5.0):
		self.DNA_size = DNA_size          # food sequence size
		self.cross_rate = cross_rate
		self.mutate_rate = mutation_rate
		self.pop_size = pop_size          # sequence pool size
		self.value_bound = value_bound    # health/taste bound
		#if pop == 'None':
		    # initialize the sequence
		    # self.pop = np.vstack([np.random.randint(0, self.value_bound, size=2) for _ in range(self.pop_size*self.DNA_size)]).reshape(self.pop_size, self.DNA_size, 2)
		target = [np.random.randint(0, self.value_bound, size=2) for _ in range(self.DNA_size)]
		self.pop = np.vstack(np.random.permutation(target) for _ in range(self.pop_size)).reshape(self.pop_size, self.DNA_size, 2)
		self.left_bound = int(DNA_size/2)
		self.right_bound = self.left_bound
		self.theta1 = theta1
		self.theta2 = theta2
		self.theta3 = theta3
		self.theta4 = theta4

	def get_fitness(self):
		fitness = np.empty((self.pop_size,), dtype=np.float64)
		leftarr = [self.pop[ind][0:ga.left_bound+1].tolist() for ind in range(ga.pop_size)]
		rightarr = [self.pop[ind][ga.right_bound:].tolist() for ind in range(ga.pop_size)]
		# print("pop", self.pop[0][0])
		cost = np.empty((self.pop_size,), dtype=np.float64)
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
			middle_weigh = popele[self.left_bound][0] + popele[self.left_bound][1]

			cost[cnt] = self.theta1*(left_taste_revcnt+right_health_revcnt) \
						+ self.theta2*(left_health_revcnt+right_taste_revcnt) \
							+ self.theta3*neighbor_difference \
								+ self.theta4*middle_weigh
		cost = pow(cost / (np.max(cost.tolist()) - np.min(cost.tolist())), 2)

		fitness = pow(self.DNA_size/mtl.sigmoid(cost, 10), 2)
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


class TravelSalesPerson(object):
	def __init__(self, n_cities):
		self.city_position = np.random.rand(n_cities, 2)
		plt.ion()

	def plotting(self, lx, ly, total_d):
		plt.cla()
		plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
		plt.plot(lx.T, ly.T, 'r-')
		plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
		plt.xlim((-0.1, 1.1))
		plt.ylim((-0.1, 1.1))
		plt.pause(0.01)


ga = GA(DNA_size=DNA_SIZE, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, value_bound=10, pop=None, theta1=1.0, theta2=0.0)
# print("first---------", ga.pop)
# env = TravelSalesPerson(N_CITIES)
best_fit_lst = []
plt.figure(1)
for generation in range(N_GENERATIONS):
	# lx, ly = ga.translateDNA(ga.pop, env.city_position)
	fitness = ga.get_fitness()
	best_idx = np.argmax(fitness)
	print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx])
	print('Best Sort:\n', ga.pop[best_idx])
	best_fit_lst.append(fitness[best_idx])
	# print("result:", ga.pop[best_idx][0:ga.left_bound].tolist())
	# print(ga.left_bound)
	ga.evolve(fitness)   

    # env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

# plt.ioff()
plt.title("Change of Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.plot(best_fit_lst)
plt.show()