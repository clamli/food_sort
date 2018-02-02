from GenericAlgorithm import GA
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import myutil as mtl
import csv

DNA_SIZE = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.01
POP_SIZE = 1000
N_GENERATIONS = 500
filename = './22_food_for_McDonald.xlsx'
column1 = 'Taste'
column2 = 'Health'

ga = GA(DNA_size=DNA_SIZE, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, value_bound=10, data='random', pop_type='default')
col_name_lst, nutrient_dict, name_lst = mtl.load_nutrient_data(filename)
target = mtl.combine_data(column1, column2, col_name_lst, nutrient_dict, plot='False')
value_target = deepcopy(target)

################ use level ####################
# taste_lst = [target[i][0] for i in range(len(target))]
# health_lst = [target[i][1] for i in range(len(target))]
# taste_lst = sorted(taste_lst)
# health_lst = sorted(health_lst)
# level = 5
# length = len(target)
# target_copy = deepcopy(target)

# for i in range(1, level+1):
# 	if i == 1:
# 		bound = int((i*length)/level)
# 		for j in range(length):
# 			if target_copy[j][0] <= taste_lst[bound]:
# 				target[j][0] = i
# 			if target_copy[j][1] <= health_lst[bound]:
# 				target[j][1] = i
# 	elif i == level:
# 		bound = int(((i-1)*length)/level)
# 		for j in range(length):
# 			if target_copy[j][0] > taste_lst[bound]:
# 				target[j][0] = i
# 			if target_copy[j][1] > health_lst[bound]:
# 				target[j][1] = i
# 	else:
# 		bound1 = int(((i-1)*length)/level)
# 		bound2 = int((i*length)/level)
# 		for j in range(length):
# 			if target_copy[j][0] > taste_lst[bound1] and target_copy[j][0] <= taste_lst[bound2]:
# 				target[j][0] = i
# 			if target_copy[j][1] > health_lst[bound1] and target_copy[j][1] <= health_lst[bound2]:
# 				target[j][1] = i
###############################################

map_dic = []
for i in range(len(target)):
	map_dic.append(target[i])


gen_seq = ga.set_data('default', target)
# ga.set_data('inverse proportional function', target)
# ga.set_data('directly proportional function', target)
ga.change_parameters(theta1=60.0, theta2=62.0, theta3=1.0, theta4=3.0, theta5=100.0, theta6=100.0, theta7=20.0, theta8=3.0, theta9=60, theta10=60)

max_fitness = 0
best_fit_lst = []
best_fit_seq = []
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
	if fitness[best_idx] not in best_fit_lst:
		best_fit_lst.append(fitness[best_idx])
		best_fit_seq.append(ga.pop[best_idx])
	if fitness[best_idx] >= max_fitness:
		max_seq = deepcopy(ga.pop[best_idx])
		max_fitness = fitness[best_idx]
	ga.evolve(fitness)
	if generation != 0:
		ax.lines.pop(0)
		ax.lines.pop(0)
	lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][0] for i in range(len(ga.pop[best_idx]))], 'r-', label="taste", marker='*')
	lines = ax.plot(range(1, len(ga.pop[best_idx])+1), [ga.pop[best_idx][i][1] for i in range(len(ga.pop[best_idx]))], 'b--', label="health", marker='*')
	plt.legend(loc = 'upper left')
	plt.pause(0.01)




output = []
max_seq = max_seq.tolist()
for i in range(len(max_seq)):
	ind = map_dic.index(max_seq[i])
	map_dic[ind] = -1
	output.append(ind)

print('target', target, '\n')
print('max_seq:', max_seq, '\n')
print('output', output, '\n')

output_name = []
output_taste = []
output_health = []
# output_taste_value = []
# output_health_value = []
for i in output:
	output_name.append(name_lst[i])
	output_taste.append(target[i][0])
	output_health.append(target[i][1])
	# output_taste_value.append(value_target[i][0])
	# output_health_value.append(value_target[i][1])
print('output_name', output_name, '\n')
print('output_taste', output_taste, '\n')
print('output_health', output_health, '\n')


row1 = ['Sort order']
row2 = ['Food name']
# row3 = ['Health level']
# row4 = ['Taste level']
row3 = ['Health value']
row4 = ['Taste value']
for i in range(len(output_name)):
	row1.append(i+1)
	row2.append(output_name[i])
	row3.append(output_health[i])
	row4.append(output_taste[i])
	# row5.append(output_health_value[i])
	# row6.append(output_taste_value[i])
with open("mcdonald_result.csv", "w", newline="") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerows([row1, row2, row3, row4])

# plt.ioff()
# plt.show()


color = ['red', 'green', 'blue', 'hotpink', 'yellow']
tmp = sorted(best_fit_lst, reverse=True)
tmp = tmp[:5]
max_seq = []
for item in tmp:
	max_seq.append(best_fit_seq[best_fit_lst.index(item)])

plt.ion()
fig = plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
plt.title("Best Sequence")
plt.xlabel("health value")
plt.ylabel("taste value")
for ind in range(len(max_seq)):
	fitness, best_seq = ga.adjust(max_seq[ind], 1, 1)
	y = [best_seq[i][0] for i in range(len(best_seq))]
	x = [best_seq[i][1] for i in range(len(best_seq))]
	plt.plot(x, y, color=color[ind], marker='*')
	plt.pause(0.01)

plt.ioff()
plt.show()


#### plot result ####
# plt.figure(1)
# plt.title("Best sequence for Nutrient dataset")
# plt.xlabel('items')
# plt.ylabel(column1 + '/' + column2)
# plt.plot(range(1, len(max_seq)+1), [max_seq[i][0] for i in range(len(max_seq))], 'r-', label=column1, marker='*')
# plt.plot(range(1, len(max_seq)+1), [max_seq[i][1] for i in range(len(max_seq))], 'b--', label=column2, marker='*')
# plt.legend(loc = 'upper left')


# plt.figure(2)
# plt.title("Initial sequence for Nutrient dataset")
# plt.xlabel('items')
# plt.ylabel('Protein (g)/Total fat (g)')
# plt.plot(range(1, len(gen_seq)+1), [gen_seq[i][0] for i in range(len(gen_seq))], 'r-', label=column1, marker='*')
# plt.plot(range(1, len(gen_seq)+1), [gen_seq[i][1] for i in range(len(gen_seq))], 'b--', label=column2, marker='*')
# plt.legend(loc = 'upper left')

# plt.figure(3)
# plt.title("Best fitness of each generation")
# plt.xlabel("Generation")
# plt.ylabel("Best Fitness")
# plt.plot(best_fit_lst)

# plt.figure(4)
# fitness, best_seq = ga.adjust(max_seq, 1.5)
# plt.title("Best taste and health trend for default dataset after fuzzy adjustment")
# plt.xlabel("item number")
# plt.ylabel("taste/health value")
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][0] for i in range(len(best_seq))], 'r-', label="Taste", marker='*')
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][1] for i in range(len(best_seq))], 'b--', label="Health", marker='*')
# plt.legend(loc = 'upper left')
# plt.show()


# fitness, best_seq = ga.adjust(max_seq, 0.5, 0.5)
# plt.figure(1)
# plt.title("Best Sequence")
# plt.xlabel("health value")
# plt.ylabel("taste value")
# y = [best_seq[i][0] for i in range(len(best_seq))]
# x = [best_seq[i][1] for i in range(len(best_seq))]
# plt.plot(x, y, marker='*')

# plt.figure(2)
# plt.title("Best taste and health trend")
# plt.xlabel("item number")
# plt.ylabel("taste/health value")
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][0] for i in range(len(best_seq))], 'r-', label="taste", marker='*')
# plt.plot(range(1, len(best_seq)+1), [best_seq[i][1] for i in range(len(best_seq))], 'b--', label="health", marker='*')
# plt.legend(loc = 'upper left')

# plt.figure(3)
# plt.title("Best taste and health trend before")
# plt.xlabel("item number")
# plt.ylabel("taste/health value")
# plt.plot(range(1, len(max_seq)+1), [max_seq[i][0] for i in range(len(max_seq))], 'r-', label="taste", marker='*')
# plt.plot(range(1, len(max_seq)+1), [max_seq[i][1] for i in range(len(max_seq))], 'b--', label="health", marker='*')
# plt.legend(loc = 'upper left')
# plt.show()