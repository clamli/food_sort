from GenericAlgorithm import GA
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

DNA_SIZE = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.01
POP_SIZE = 500
N_GENERATIONS = 300

ga = GA(DNA_size=DNA_SIZE, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, value_bound=10, data='random', pop_type='default')
max_seq1 = []

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
    if fitness[best_idx] >= max_fitness:
        max_seq1 = deepcopy(ga.pop[best_idx])
        max_fitness = fitness[best_idx]
    ga.evolve(fitness)
    if generation != 0:
	    ax.lines.pop(0)
	    ax.lines.pop(0)
    lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][0] for i in range(len(ga.pop[best_idx]))], 'r-', label="taste", marker='*')
    lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][1] for i in range(len(ga.pop[best_idx]))], 'b--', label="health", marker='*')
    plt.legend(loc = 'upper left')
    plt.pause(0.01)
    
plt.ioff()
plt.show()

# plt.figure(1)
# plt.title("Best taste and health trend for default dataset")
# plt.xlabel("item number")
# plt.ylabel("taste/health value")
# plt.plot(range(1, len(max_seq1)+1), [max_seq1[i][0] for i in range(len(max_seq1))], 'r-', label="taste", marker='*')
# plt.plot(range(1, len(max_seq1)+1), [max_seq1[i][1] for i in range(len(max_seq1))], 'b--', label="health", marker='*')
# plt.legend(loc = 'upper left')


# plt.figure(2)
# fitness, best_seq = ga.adjust(max_seq1)
# plt.title("Best taste and health trend for default dataset after fuzzy adjustment")
# plt.xlabel("item number")
# plt.ylabel("taste/health value")
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][0] for i in range(len(best_seq))], 'r-', label="taste", marker='*')
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][1] for i in range(len(best_seq))], 'b--', label="health", marker='*')
# plt.legend(loc = 'upper left')
# plt.show()

fitness, best_seq = ga.adjust(max_seq1)
plt.figure(1)
plt.title("Best Sequence")
plt.xlabel("health value")
plt.ylabel("taste value")
y = [best_seq[i][0] for i in range(len(best_seq))]
x = [best_seq[i][1] for i in range(len(best_seq))]
plt.plot(x, y, marker='*')

plt.figure(2)
plt.title("Best taste and health trend")
plt.xlabel("item number")
plt.ylabel("taste/health value")
plt.plot(range(1, len(best_seq)+1), [best_seq[i][0] for i in range(len(best_seq))], 'r-', label="taste", marker='*')
plt.plot(range(1, len(best_seq)+1), [best_seq[i][1] for i in range(len(best_seq))], 'b--', label="health", marker='*')
plt.legend(loc = 'upper left')
plt.show()

