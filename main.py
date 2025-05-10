import math
from math import log2

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

import networkx as nx

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

n = 6
vertices = list(range(3*n))

if n <= 2:
    num_v = 9
else:
    num_v = 6 * math.ceil(log2(n))

model = LpProblem("Clique_Covering", LpMinimize)

vert = LpVariable.dicts("x", (vertices, range(num_v)), cat="Binary")

model += 0
num_v = 13

edges = LpVariable.dicts("z", (vertices, vertices, range(num_v)), cat="Binary")

m = LpVariable.dicts("m", (range(n), range(num_v)), cat="Binary") # m[i][j] = 1 - минимальное i, что vert[i][j] != 1



for i in range(0, len(vertices), 3):
    model += (vert[i][0] == 1)


for i in range(n):
    for j in range(num_v):
        model += (m[i][j] <= 1 - vert[i*3][j])
        for k in range(j):
            model += (m[i][j] <= vert[i*3][k])

for i in vertices:
    for j in vertices:
        for k in range(num_v):
            if i < j and i // 3 != j // 3:
                model += edges[i][j][k] <= vert[i][k]
                model += edges[i][j][k] <= vert[j][k]
                model += edges[i][j][k] >= vert[i][k] + vert[j][k] - 1

for k in range(num_v - 1):
    for i in range(n):
        model += m[i][k] >= m[i][k + 1]

for k in range(num_v):
    for i in range(0, len(vertices), 3):
        model += vert[i][k] + vert[i + 1][k] + vert[i + 2][k] == 1

for k in range(num_v):
    for i in range(n):
        model += vert[i*3 + 2][k] <= 1 - m[i][k]

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
    #     for j in graph[i]:
    #         for k in range(num_v):
    #             if x[i][k].varValue == 1 and x[j][k].varValue == 1 and i < j:
    #                 print(f"Edge {i}-{j} is covered by Clique {k + 1}.")



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
