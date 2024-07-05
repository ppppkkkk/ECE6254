import gurobipy as gb

model = gb.Model("Stochastic_Static_G&TEP")

model.setParam('MIPGap', 0.05)

operation_points = ['o1', 'o2']
work_points = ['w1', 'w2']
demands = {('o1', 'w1'): 203, ('o2', 'w1'): 385, ('o1', 'w2'): 377, ('o2', 'w2'): 715}

pc_max = model.addVar(lb=0, ub=500, name='pc_max')
x_l = model.addVar(vtype=gb.GRB.BINARY, name='x_l')
pg = model.addVars(operation_points, work_points, lb=0, ub=400, name='pg')
pc = model.addVars(operation_points, work_points, lb=0, name='pc')
pl1 = model.addVars(operation_points, work_points, lb=-200, ub=200, name='pl1')
pl2 = model.addVars(operation_points, work_points, lb=-200, ub=200, name='pl2')

model.update()

model.addConstr(700000 * pc_max <= 400000000)
model.addConstr(1000000 * x_l <= 2000000)
for o in operation_points:
    for w in work_points:
        model.addConstr(pg[o, w] - pl1[o, w] - pl2[o, w] == 0)
        model.addConstr(pc[o, w] + pl1[o, w] + pl2[o, w] >= demands[o, w])
        model.addConstr(pl2[o, w] >= -200 * x_l)
        model.addConstr(pl2[o, w] <= 200 * x_l)
        model.addConstr(pc[o, w] <= pc_max)

investment_cost = 70000 * pc_max + 100000 * x_l
operational_cost = gb.quicksum(
    0.5 * (6000 * (35 * pg['o1', w] + 25 * pc['o1', w]) + 2760 * (35 * pg['o2', w] + 25 * pc['o2', w]))
    for w in work_points
)
model.setObjective(investment_cost + operational_cost, gb.GRB.MINIMIZE)

model.optimize()

if model.status == gb.GRB.OPTIMAL:
    for var in model.getVars():
        print(f'{var.VarName} = {var.X}')
    print(f'Objective = {model.ObjVal}')
else:
    print('No solution found')


