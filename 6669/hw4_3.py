# import gurobipy as gb
# from gurobipy import GRB
# model = gb.Model("Stochastic_Dynamic_G&TEP")
# model.setParam('MIPGap', .05)
# operators = ['o1', 'o2']
# time_periods = ['t1', 't2']
# wind_scenarios = ['w1', 'w2', 'w3', 'w4']
# demand = {('o1', 't1', 'w1'): 212, ('o1', 't2', 'w1'): 214,
#     ('o2', 't1', 'w1'): 402, ('o2', 't2', 'w1'): 407,
#     ('o1', 't1', 'w2'): 212, ('o1', 't2', 'w2'): 284,
#     ('o2', 't1', 'w2'): 402, ('o2', 't2', 'w2'): 539,
#     ('o1', 't1', 'w3'): 281, ('o1', 't2', 'w3'): 284,
#     ('o2', 't1', 'w3'): 533, ('o2', 't2', 'w3'): 539,
#     ('o1', 't1', 'w4'): 281, ('o1', 't2', 'w4'): 377,
#     ('o2', 't1', 'w4'): 533, ('o2', 't2', 'w4'): 715,
# }
#
# pcmax = model.addVars(time_periods, wind_scenarios, lb=0, ub=500, name='pcmax')
# xl = model.addVars(time_periods, wind_scenarios, vtype=GRB.BINARY, name='xl')
# pg = model.addVars(operators, time_periods, wind_scenarios, lb=0, ub=400, name='pg')
# pc = model.addVars(operators, time_periods, wind_scenarios, lb=0, name='pc')
# pl1 = model.addVars(operators, time_periods, wind_scenarios, lb=-200, ub=200, name='pl1')
# pl2 = model.addVars(operators, time_periods, wind_scenarios, lb=-200, ub=200, name='pl2')
# model.update()
#
# model.addConstrs(700000 * pcmax[t, w] <= 400000000 for t in time_periods for w in wind_scenarios)
# model.addConstrs(1000000 * xl[t, w] <= 2000000 for t in time_periods for w in wind_scenarios)
#
# model.addConstrs(gb.quicksum(pcmax[t, w] for t in time_periods) <= 500 for w in wind_scenarios)
# model.addConstrs(gb.quicksum(xl[t, w] for t in time_periods) <= 1 for w in wind_scenarios)
# model.addConstrs(pg[o, t, w] - pl1[o, t, w] - pl2[o, t, w] == 0 for o in operators for t in time_periods for w in wind_scenarios)
# model.addConstrs(pc[o, t, w] + pl1[o, t, w] + pl2[o, t, w] >= demand[o, t, w] for o in operators for t in time_periods for w in wind_scenarios)
#
# model.addConstrs(pl2[o, 't1', w] >= -200 * xl['t1', w] for o in operators for w in wind_scenarios)
# model.addConstrs(pl2[o, 't1', w] <= 200 * xl['t1', w] for o in operators for w in wind_scenarios)
# model.addConstrs(pc[o, 't1', w] <= pcmax['t1', w] for o in operators for w in wind_scenarios)
# model.addConstrs(pl2[o, 't2', w] >= -200 * gb.quicksum(xl[t, w] for t in time_periods) for o in operators for w in wind_scenarios)
# model.addConstrs(pl2[o, 't2', w] <= 200 * gb.quicksum(xl[t, w] for t in time_periods) for o in operators for w in wind_scenarios)
# model.addConstrs(pc[o, 't2', w] <= gb.quicksum(pcmax[t, w] for t in time_periods) for o in operators for w in wind_scenarios)
#
# model.addConstrs(pcmax['t1', w] == pcmax['t1', 'w1'] for w in wind_scenarios)
# model.addConstrs(pcmax['t2', 'w1'] == pcmax['t2', 'w2'] for w in wind_scenarios)
# model.addConstrs(pcmax['t2', 'w3'] == pcmax['t2', 'w4'] for w in wind_scenarios)
# model.addConstrs(xl['t1', w] == xl['t1', 'w1'] for w in wind_scenarios)
# model.addConstrs(xl['t2', 'w1'] == xl['t2', 'w2'] for w in wind_scenarios)
# model.addConstrs(xl['t2', 'w3'] == xl['t2', 'w4'] for w in wind_scenarios)
#
# investment_cost = gb.quicksum(0.25 * (140000 * pcmax['t1', w] + 70000 * pcmax['t2', w] + 200000 * xl['t1', w] + 100000 * xl['t2', w]) for w in wind_scenarios)
# operation_cost = gb.quicksum(0.25 * (6000 * (35 * pg['o1', t, w] + 25 * pc['o1', t, w]) + 2760 * (35 * pg['o2', t, w] + 25 * pc['o2', t, w])) for t in time_periods for w in wind_scenarios)
# model.setObjective(investment_cost + operation_cost, gb.GRB.MINIMIZE)
# model.setParam( 'OutputFlag', False)
# model.optimize()
# if model.status == GRB.OPTIMAL:
#     for v in model.getVars():
#         print('%s = %g' % (v.VarName, v.X))
#         print('0bj = %g' % model.ObjVal)
#     else:
#         print('NO SOLUTION')


import matplotlib.pyplot as plt
import numpy as np

# Sample data from the user's model output for demonstration purposes
# This would be replaced by the actual data extracted from the Gurobi model
operators = ['o1', 'o2']
time_periods = ['t1', 't2']
wind_scenarios = ['w1', 'w2', 'w3', 'w4']
pg_values = np.random.randint(0, 400, size=(len(operators), len(time_periods), len(wind_scenarios)))
pl1_values = np.random.randint(-200, 200, size=(len(operators), len(time_periods), len(wind_scenarios)))
pl2_values = np.random.randint(-200, 200, size=(len(operators), len(time_periods), len(wind_scenarios)))
xl_values = np.random.randint(0, 2, size=(len(time_periods), len(wind_scenarios)))
print(pg_values)
# We'll create a grid of subplots, with one subplot for each time period and wind scenario
fig, axs = plt.subplots(len(time_periods), len(wind_scenarios), figsize=(15, 5), constrained_layout=True)

# Set a title for each column which corresponds to a wind scenario
for ax, scenario in zip(axs[0], wind_scenarios):
    ax.set_title(f'Wind Scenario {scenario}')

# Set a label for each row which corresponds to a time period
for ax, period in zip(axs[:, 0], time_periods):
    ax.set_ylabel(f'Time {period}', rotation=90, size='large')

# Plotting the power flow for each operator in each time period and wind scenario
for i, t in enumerate(time_periods):
    for j, w in enumerate(wind_scenarios):
        ax = axs[i, j]
        index = (slice(None), i, j)  # slice for the current time period and wind scenario
        bars = ax.bar(operators, pg_values[index], label='Generation (pg)')

        # We use hlines to represent the power line flow
        for operator_index, operator in enumerate(operators):
            ax.hlines(y=pg_values[operator_index, i, j], xmin=-0.4 + operator_index, xmax=0.4 + operator_index,
                      color='orange' if pl1_values[operator_index, i, j] > 0 else 'blue',
                      linestyles='dashed', label='Line 1 Power (pl1)' if operator_index == 0 else "")
            ax.hlines(y=pg_values[operator_index, i, j], xmin=-0.4 + operator_index, xmax=0.4 + operator_index,
                      color='green' if pl2_values[operator_index, i, j] > 0 else 'red',
                      linestyles='dotted', label='Line 2 Power (pl2)' if operator_index == 0 else "")

        # Show the line existence as a text label
        ax.text(1.1, ax.get_ylim()[1] * 0.9, f'Line Exist: {bool(xl_values[i, j])}', verticalalignment='center')

# Create a legend for the first subplot only to avoid repetition
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)

# Adjust layout
plt.tight_layout(pad=3.0)

# Display the plot
plt.show()
