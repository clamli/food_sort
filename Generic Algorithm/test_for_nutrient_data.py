from GenericAlgorithm import GA
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import myutil as mtl

DNA_SIZE = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.01
POP_SIZE = 500
N_GENERATIONS = 300
filename = './100_foods_calorie_and_core_nutrients.xlsx'
column1 = 'Total fat (g)'
column2 = 'Dietary fibre (g)'

ga = GA(DNA_size=DNA_SIZE, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, value_bound=10, data='random', pop_type='default')
col_name_lst, nutrient_dict = mtl.load_nutrient_data(filename)
target = mtl.combine_data(column1, column2, col_name_lst, nutrient_dict, plot='False')
gen_seq = ga.set_data('default', target[0:20])
# ga.set_data('inverse proportional function', target)
# ga.set_data('directly proportional function', target)
ga.change_parameters(theta1=60.0, theta2=62.0, theta3=1.0, theta4=3.0, theta5=10.0, theta6=10.0, theta7=1.0, theta8=3.0, theta9=30, theta10=75)

max_fitness = 0
best_fit_lst = []
# ret_result = ga.generate_init_data('default', ga.target)
# ret_result = ga.generate_init_data('inverse proportional function', ga.target)
# ret_result = ga.generate_init_data('directly proportional function', ga.target)

plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
plt.title("Taste and health trend for default dataset")
plt.xlabel("item number")
plt.ylabel("taste/health value")
#### for default target ####
max_fitness = 0
for generation in range(N_GENERATIONS):
    fitness = ga.get_fitness()
    best_idx = np.argmax(fitness)
    # print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx])
    # print('Best Sort:\n', ga.pop[best_idx])
    best_fit_lst.append(fitness[best_idx])
    if fitness[best_idx] >= max_fitness:
        max_seq = deepcopy(ga.pop[best_idx])
        max_fitness = fitness[best_idx]
    ga.evolve(fitness)
    if generation != 0:
	    ax.lines.pop(0)
	    ax.lines.pop(0)
    lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][0] for i in range(len(ga.pop[best_idx]))], 'r-', label=column1, marker='*')
    lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][1] for i in range(len(ga.pop[best_idx]))], 'b--', label=column2, marker='*')
    plt.legend(loc = 'upper left')
    plt.pause(0.01)

    
plt.ioff()
plt.show()


#### plot result ####
plt.figure(1)
plt.title("Best sequence for Nutrient dataset")
plt.xlabel('items')
plt.ylabel(column1 + '/' + column2)
plt.plot(range(1, len(max_seq)+1), [max_seq[i][0] for i in range(len(max_seq))], 'r-', label=column1, marker='*')
plt.plot(range(1, len(max_seq)+1), [max_seq[i][1] for i in range(len(max_seq))], 'b--', label=column2, marker='*')
plt.legend(loc = 'upper left')


plt.figure(2)
plt.title("Initial sequence for Nutrient dataset")
plt.xlabel('items')
plt.ylabel('Protein (g)/Total fat (g)')
plt.plot(range(1, len(gen_seq)+1), [gen_seq[i][0] for i in range(len(gen_seq))], 'r-', label=column1, marker='*')
plt.plot(range(1, len(gen_seq)+1), [gen_seq[i][1] for i in range(len(gen_seq))], 'b--', label=column2, marker='*')
plt.legend(loc = 'upper left')

plt.figure(3)
plt.title("Best fitness of each generation")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.plot(best_fit_lst)

plt.show()