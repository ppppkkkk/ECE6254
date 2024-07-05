import pandas as pd
import gurobipy as gp
from gurobipy import GRB


file_path = "bonddata.xlsx"


data = pd.read_excel(file_path, header=3)


periods, bonds = 14, 7


rt = data.iloc[0:periods, 1].tolist()
Lt = data.iloc[0:periods, 2].tolist()
Pi = data.iloc[0, 3:3+bonds].tolist()

Cit = [data.iloc[0:periods, 3+i].tolist() for i in range(bonds)]

model = gp.Model("bond_portfolio")

x = model.addVars(bonds, name="x")
z = model.addVars(periods+1, name="z")

for t in range(periods):
    model.addConstr(sum(Cit[i][t] * x[i] for i in range(bonds)) + z[t] * (1 + rt[t]) == Lt[t] + z[t+1])

model.addConstr(sum(Pi[i] * x[i] for i in range(bonds)) == z[0])

model.setObjective(z[0], GRB.MINIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal initial cash investment: {model.objVal:.2f}")
    for i in range(bonds):
        print(f"Units of bond {i+1} to purchase: {x[i].X:.2f}")
    for t in range(periods+1):
        print(f"Cash balance at the end of period {t}: {z[t].X:.2f}")
else:
    print("No optimal solution found.")
