import math
from math import log2

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
# num_v = (3*n)*(3*n - 1)//2 - 3*n

num_v = 6 * math.ceil(log2(n))
print(num_v)

model = LpProblem("Clique_Covering", LpMinimize)

x = LpVariable.dicts("x", (vertices, range(num_v)), cat="Binary")
y = LpVariable.dicts("y", range(num_v), cat="Binary")

model += lpSum(y[k] for k in range(num_v))

z = LpVariable.dicts("z", (vertices, vertices, range(num_v)), cat="Binary")

#model += x[0][0] == 1

for i in vertices:
    for j in vertices:
        for k in range(num_v):
            if i < j:
                model += z[i][j][k] <= x[i][k]
                model += z[i][j][k] <= x[j][k]
                model += (z[i][j][k] >= x[i][k] + x[j][k] - 1)

for k in range(num_v):
    for u in vertices:
        for v in vertices:
            if u < v and v not in graph[u]:
                model += z[u][v][k] == 0 #x[u][k] + x[v][k] <= 1 #z[u][v][k] == 0

for i in vertices:
    for k in range(num_v):
        model += x[i][k] <= y[k]

for i in vertices:
    for j in graph[i]:
        if i < j:
            model += lpSum(z[i][j][k] for k in range(num_v)) >= 1



model.solve()

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
# первую клику
# на git
