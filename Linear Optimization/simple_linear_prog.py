from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('my_LP', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

x = solver.NumVar(0, 10, 'my first variable')
y = solver.NumVar(0, 10, 'my second variable')

solver.Add(x + y <= 7)
solver.Add(x - 2 * y >= 2)

objective = solver.Objective()
objective.SetCoefficient(x, 3)
objective.SetCoefficient(y, 1)
objective.SetMaximization()

status = solver.Solve()

if status not in [solver.OPTIMAL, solver.FEASIBLE]:
	raise Exception('Unable to find feasible solution')

print(x.solution_value())
print(y.solution_value())