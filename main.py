import math
from math import log2

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

import networkx as nx

def generate_triangle_free_graph(n):
    total_nodes = 3 * n
    G = nx.complete_graph(total_nodes)

    vertices = list(range(total_nodes))

    for i in range(n):
        u, v, w = vertices[3 * i:3 * i + 3]
        G.remove_edge(u, v)
        G.remove_edge(v, w)
        G.remove_edge(w, u)

    return G




n = 5
graph = generate_triangle_free_graph(n)
vertices = list(range(3*n))

if n <= 2:
    num_v = 9
else:
    num_v = 6 * math.ceil(log2(n))

model = LpProblem("Clique_Covering", LpMinimize)

vert = LpVariable.dicts("x", (vertices, range(num_v)), cat="Binary")
y = LpVariable.dicts("y", range(num_v), cat="Binary")

model += 0
model += lpSum(y[k] for k in range(num_v)) == 8

edges = LpVariable.dicts("z", (vertices, vertices, range(num_v)), cat="Binary")

m = LpVariable.dicts("m", (range(n), range(num_v)), cat="Binary")

s = LpVariable.dicts("s", (range(n), range(num_v)), cat="Binary")


for i in range(0, len(vertices), 3):
    model += (vert[i][0] == 1)

for i in range(0, n - 1):
    for k in range(num_v):
        model += s[i][k] >= vert[3 * i][k]
        model += s[i][k] >= vert[3 * i + 1][k]
        model += s[i][k] >= vert[3 * i + 2][k]
        model += s[i][k] <= vert[3 * i][k] + vert[3 * i + 1][k] + vert[3 * i + 2][k]
for i in range(n - 1):
    lhs = lpSum(2**(num_v - 1 - t) * s[i][t] for t in range(num_v))
    rhs = lpSum(2**(num_v - 1 - t) * s[i+1][t] for t in range(num_v))
    model += lhs >= rhs

for i in range(n):
    for j in range(num_v):
        model += (m[n][j] <= 1 - vert[i*3][j])
        for k in range(j):
            model += (m[n][j] <= vert[i*3][k])

for i in vertices:
    for j in vertices:
        for k in range(num_v):
            if i < j:
                model += edges[i][j][k] <= vert[i][k]
                model += edges[i][j][k] <= vert[j][k]
                model += (edges[i][j][k] >= vert[i][k] + vert[j][k] - 1)


if len(vertices) > 5:
    model += edges[0][3][0] == 1
    model += edges[0][4][1] == 1
    model += edges[0][5][2] == 1

for k in range(num_v):
    for i in range(0, len(vertices), 3):
        model += vert[i][k] + vert[i + 1][k] + vert[i + 2][k] <= 1

for k in range(num_v):
    for i in range(n):
        model += vert[i*3 + 2][k] <= 1 - m[i][k]

for i in vertices:
    for k in range(num_v):
        model += vert[i][k] <= y[k]

for i in vertices:
    for j in graph[i]:
        if i < j:
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

    count = 0
    for k in range(num_v):
        if y[k].varValue == 1:
            count+=1
    print(count)

    # for k in range(num_v):
    #     for v in vertices:
    #         print(f"x__{v}_{k} = {x[v][k].varValue}")
    # for k in range(7, 8):
    #     for i in vertices:
    #         for j in vertices:
    #             if i < j:
    #                 print(x[i][k].varValue + x[j][k].varValue - 1)
    #                 print(f"z__{i}_{j}_{k} = {z[i][j][k].varValue}")
    #
    # for name, constraint in model.constraints.items():
    #     print(f"{name}: {constraint}")


else:
    print("No solution found.")


# checker

