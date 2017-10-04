class DietOptimizer(object):
	def __init__(self, nutrient_data_filename='nutrients.csv',
						nutrient_constraints_filename='constraints.csv'):

		self.food_table = 
		self.constraints_data = 

		self.solver = pywraplp.Solver('diet_optimizer', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
		self.create_variable_dict()
		self.create_constraints()

		self.objective = self.solver.Objective()
		for row in self.food_table:
			name = row['description']
			var = self.variable_dict[name]
			calories_in_food = row[calories_name]
			self.objective.SetCoefficient(var, calories_in_food)
		self.objective.SetMinimization()

	def solve(self):
		'''
			Return a dictionary with 'foods' and 'nutrients' keys representing
			the solution and the nutrient amounts for the chosen diet
		'''
		status = self.solver.Solve()
		if status not in [self.solver.OPTIMAL, self.solver.FEASIBLE]:
			raise Exception('Unable to find feasible solution')

		chosen_foods = {
			food_name: var.solution_value()
			for food_name, var in self.variable_dict.items() if var.solution_value() > 1e-10
		}

		self.chosen_foods = chosen_foods

		nutrients = {
			row['nutrient']: self.nutrients_in_diet(chosen_foods, row['nutrient'])
			for row in self.constraints_table
		}

		return {
			'food': chosen_foods,
			'nutrients': nutrients,
		}