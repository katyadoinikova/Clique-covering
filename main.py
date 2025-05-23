import math
from math import log2

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

def check_solution(x):
    for k in range(num_v):
        for i in range(3*n):
            for j in range(i + 1, 3*n):
                if x[i][k].varValue == 1 and x[j][k].varValue == 1:
                    assert (i // 3 != j // 3)
    for i in range(3 * n):
        for j in range(i + 1, 3 * n):
            if i // 3 != j // 3:
                is_covered = False
                for k in range(num_v):
                    if x[i][k].varValue == 1 and x[j][k].varValue == 1:
                        is_covered = True
                        break
                assert is_covered
    print("Checked")

n = 10
vertices = list(range(3*n))

# if n <= 2:
#     num_v = 9
# else:
#     num_v = 6 * math.ceil(log2(n))

num_v = 20

model = LpProblem("Clique_Covering", LpMinimize)

vert = LpVariable.dicts("x", (vertices, range(num_v)), cat="Binary")


edges = LpVariable.dicts("z", (vertices, vertices, range(num_v)), cat="Binary")

min_in_share = LpVariable.dicts("m", (range(n), range(num_v)), cat="Binary") # m[i][j] = 1 - до j+1, что j - первый 0 у vert[3*i][j]
model += 0

for i in range(0, len(vertices), 3):
    model += (vert[i][0] == 1)

for i in range(3):
    for j in range(3):
        model += edges[i][j + 3][3*i + j] == 1
model += edges[0][3][9] == 1

for i in vertices:
    for j in vertices:
        for k in range(num_v):
            if i < j and i // 3 != j // 3:
                model += edges[i][j][k] <= vert[i][k]
                model += edges[i][j][k] <= vert[j][k]
                model += edges[i][j][k] >= vert[i][k] + vert[j][k] - 1


for k in range(num_v):
    for i in range(0, len(vertices), 3):
        model += vert[i][k] + vert[i + 1][k] + vert[i + 2][k] == 1

for i in range(n):
    model += min_in_share[i][0] == 1
    for k in range(num_v - 1):
        model += min_in_share[i][k+1] <= min_in_share[i][k]
        model += min_in_share[i][k+1] <= vert[3*i][k]
        model += min_in_share[i][k+1] >= min_in_share[i][k] + vert[3*i][k] - 1


for k in range(num_v):
    for i in range(n):
        model += vert[i*3 + 2][k] <= 1 - min_in_share[i][k]

eq = LpVariable.dicts("eq", (range(n-1), range(num_v)), cat="Binary") #x[3*i][k]== x[3*(i+1)][k]?

for i in range(n - 1):
    for k in range(num_v):
        model += eq[i][k] <= 1 - vert[3*i][k] + vert[3*(i+1)][k]
        model += eq[i][k] <= 1 + vert[3*i][k] - vert[3*(i+1)][k]
        model += eq[i][k] >= vert[3*i][k] + vert[3*(i+1)][k] - 1
        model += eq[i][k] >= 1 - vert[3*i][k] - vert[3*(i+1)][k]


u = LpVariable.dicts("u", (range(n-1), range(num_v)), cat="Binary")  # u[i][k] = 1 ⇔ все позиции до k равны
gt = LpVariable.dicts("gt", (range(n-1), range(num_v)), cat="Binary")  # gt[i][k] = 1 ⇔ x[3*i][k]=1, x[3*(i+1)][k]=0 и до k все равны
for i in range(n - 1):
    model += u[i][0] == 1
    for k in range(2, num_v - 1):
        model += u[i][k+1] <= u[i][k]
        model += u[i][k+1] <= eq[i][k]
        model += u[i][k+1] >= u[i][k] - 1 + eq[i][k]
for i in range(n - 1):
    for k in range(2, num_v):
        model += gt[i][k] <= u[i][k]
        model += gt[i][k] <= vert[3*i][k]
        model += gt[i][k] <= 1 - vert[3*(i+1)][k]
        model += gt[i][k] >= u[i][k] + vert[3*i][k] - vert[3*(i+1)][k] - 1
for i in range(n - 1):
    model += lpSum(gt[i][k] for k in range(2, num_v)) >= 1



for i in range(3 * n):
    for j in range(i + 1, 3 * n):
        if i // 3 != j // 3:
            model += lpSum(edges[i][j][k] for k in range(num_v)) >= 1


print(pulp.listSolvers(onlyAvailable=True))
model.solve(pulp.getSolver('SCIP_PY'))

if model.status == 1:
    print("Solution found:")
    # for k in range(num_v):
    #     if y[k].varValue == 1:
    #         print(f"Clique {k + 1} is chosen.")
    # for i in vertices:
    #     for j in vertices:
    #         for k in range(num_v):
    #             if i < j and edges[i][j][k].varValue == 1:
    #                 print(f"Edge {i}-{j} is covered by Clique {k + 1}.")
    # for k in range(num_v):
    #     print(edges[0][7][k].varValue)
                # if vert[i][k].varValue == 1 and vert[j][k].varValue == 1 and i < j:
                #     print(f"Edge {i}-{j} is covered by Clique {k + 1}.")



    print(num_v)
    check_solution(vert)

    # for k in range(num_v):
    #     for v in vertices:
    #         print(f"x__{v}_{k} = {vert[v][k].varValue}")
    # for k in range(7, 8):
    #     for i in vertices:
    #         for j in vertices:
    #             if i < j:
    #                 print(x[i][k].varValue + vert[j][k].varValue - 1)
    #                 print(f"z__{i}_{j}_{k} = {z[i][j][k].varValue}")
    #
    # for name, constraint in model.constraints.items():
    #     print(f"{name}: {constraint}")


else:
    print("No solution found.")








