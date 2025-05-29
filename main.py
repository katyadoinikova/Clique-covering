from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

def binary_round(x, tol=1e-6):
    if abs(x - 1.0) < tol:
        return 1
    elif abs(x) < tol:
        return 0
    else:
        raise ValueError(f"Значение {x} не близко ни к 0, ни к 1!")

def check_solution(x):
    for k in range(numCliques):
        for i in range(3*n):
            for j in range(i + 1, 3*n):
                if binary_round(x[i][k].varValue) == 1 and binary_round(x[j][k].varValue) == 1:
                    assert (i // 3 != j // 3)
    for i in range(3 * n):
        for j in range(i + 1, 3 * n):
            if i // 3 != j // 3:
                is_covered = False
                for k in range(numCliques):
                    if binary_round(x[i][k].varValue) == 1 and binary_round(x[j][k].varValue) == 1:
                        is_covered = True
                        break
                if (not(is_covered)):
                    print(i, j)
                    for k in range(numCliques):
                        print(x[1][k].varValue, x[6][k].varValue)

                assert is_covered
    print("Checked")

n = 8
vertices = list(range(3*n))

# if n <= 2:
#     numCliques = 9
# else:
#     numCliques = 6 * math.ceil(log2(n))

numCliques = 11

model = LpProblem("Clique_Covering", LpMinimize)

vert = LpVariable.dicts("x", (vertices, range(numCliques)), cat="Binary")


edges = LpVariable.dicts("z", (vertices, vertices, range(numCliques)), cat="Binary")

model += 0

for i in range(0, len(vertices), 3):
    model += (vert[i][0] == 1)

for k in range(numCliques):
    for i in range(0, len(vertices), 3):
        model += vert[i][k] + vert[i + 1][k] + vert[i + 2][k] == 1


for i in vertices:
    for j in vertices:
        for k in range(numCliques):
            if i < j and i // 3 != j // 3:
                model += edges[i][j][k] <= vert[i][k]
                model += edges[i][j][k] <= vert[j][k]
                model += edges[i][j][k] >= vert[i][k] + vert[j][k] - 1

if numCliques > 9:
    for i in range(3):
        for j in range(3):
            model += edges[i][j + 3][3 * i + j] == 1
    model += edges[0][3][9] == 1

first_val = LpVariable.dicts("first_val", range(10, numCliques), cat="Integer")

for k in range(10, numCliques):

    model += first_val[k] == lpSum(i * vert[i][k]  for i in range(3))


for k in range(10, numCliques - 1):
    model += first_val[k] <= first_val[k + 1]
    


model = LpProblem("Clique_Covering", LpMinimize)

vert = LpVariable.dicts("x", (vertices, range(numCliques)), cat="Binary")


edges = LpVariable.dicts("z", (vertices, vertices, range(numCliques)), cat="Binary")

model += 0

for i in range(0, len(vertices), 3):
    model += (vert[i][0] == 1)

for k in range(numCliques):
    for i in range(0, len(vertices), 3):
        model += vert[i][k] + vert[i + 1][k] + vert[i + 2][k] == 1


for i in vertices:
    for j in vertices:
        for k in range(numCliques):
            if i < j and i // 3 != j // 3:
                model += edges[i][j][k] <= vert[i][k]
                model += edges[i][j][k] <= vert[j][k]
                model += edges[i][j][k] >= vert[i][k] + vert[j][k] - 1

if numCliques > 9:
    for i in range(3):
        for j in range(3):
            model += edges[i][j + 3][3 * i + j] == 1
    model += edges[0][3][9] == 1

first_val = LpVariable.dicts("first_val", range(10, numCliques), cat="Integer")

for k in range(10, numCliques):
    model += first_val[k] == lpSum(i * vert[i][k]  for i in range(3))
for k in range(10, numCliques - 1):
    model += first_val[k] <= first_val[k + 1]

sortCliques = LpVariable.dicts("sort_cliques", range(10, numCliques), cat="Integer")

for k in range(10, numCliques):
    model += sortCliques[k] == lpSum(2 ** (3 * n - i) * vert[i][k] for i in range(3 * n))
for k in range(10, numCliques - 1):
    model += sortCliques[k] >= sortCliques[k + 1]

sortTriang = LpVariable.dicts("sort_triang", range(2, n), cat="Integer")

for i in range(2, n):
    model += sortTriang[i] == lpSum(2 ** (numCliques - k) * vert[3 * i][k] for k in range(numCliques))
for i in range(2, n-1):
    model += sortTriang[i] >= sortTriang[i + 1]


min_in_share = LpVariable.dicts("m", (range(n), range(numCliques)), cat="Binary") # m[i][j] = 1 - до j+1, что j - первый 0 у vert[3*i][j]

for i in range(n):
    model += min_in_share[i][0] == 1
    for k in range(numCliques - 1):
        model += min_in_share[i][k+1] <= min_in_share[i][k]
        model += min_in_share[i][k+1] <= vert[3*i][k]
        model += min_in_share[i][k+1] >= min_in_share[i][k] + vert[3*i][k] - 1


for k in range(numCliques):
    for i in range(n):
        model += vert[i*3 + 2][k] <= 1 - min_in_share[i][k]

eq = LpVariable.dicts("eq", (range(n-1), range(numCliques)), cat="Binary") #x[3*i][k]== x[3*(i+1)][k]?

for i in range(n - 1):
    for k in range(numCliques):
        model += eq[i][k] <= 1 - vert[3*i][k] + vert[3*(i+1)][k]
        model += eq[i][k] <= 1 + vert[3*i][k] - vert[3*(i+1)][k]
        model += eq[i][k] >= vert[3*i][k] + vert[3*(i+1)][k] - 1
        model += eq[i][k] >= 1 - vert[3*i][k] - vert[3*(i+1)][k]


u = LpVariable.dicts("u", (range(2, n-1), range(numCliques)), cat="Binary")  # u[i][k] = 1 ⇔ все позиции до k-1 равны
gt = LpVariable.dicts("gt", (range(2, n-1), range(numCliques)), cat="Binary")  # gt[i][k] = 1 ⇔ x[3*i][k]=1, x[3*(i+1)][k]=0 и до k все равны
for i in range(2, n - 1):
    model += u[i][0] == 1
    for k in range(numCliques - 1):
        model += u[i][k+1] <= u[i][k]
        model += u[i][k+1] <= eq[i][k]
        model += u[i][k+1] >= u[i][k] - 1 + eq[i][k]
for i in range(2, n - 1):
    for k in range(numCliques):
        model += gt[i][k] <= u[i][k]
        model += gt[i][k] <= vert[3*i][k]
        model += gt[i][k] <= 1 - vert[3*(i+1)][k]
        model += gt[i][k] >= u[i][k] + vert[3*i][k] - vert[3*(i+1)][k] - 1
for i in range(2, n - 1):
    model += lpSum(gt[i][k] for k in range(numCliques)) >= 1






eq1 = LpVariable.dicts("eq1", (range(3*n), range(9, numCliques-1)), cat="Binary") #x[i][k]== x[i][k+1]?

for k in range(9, numCliques-1):
    for i in range(3*n):
        model += eq1[i][k] <= 1 - vert[i][k] + vert[i][k+1]
        model += eq1[i][k] <= 1 + vert[i][k] - vert[i][k+1]
        model += eq1[i][k] >= vert[i][k] + vert[i][k+1] - 1
        model += eq1[i][k] >= 1 - vert[i][k] - vert[i][k+1]


u1 = LpVariable.dicts("u1", (range(3*n), range(9, numCliques-1)), cat="Binary")  # u1[i][k] = 1 ⇔ все позиции до i-1 равны
gt1 = LpVariable.dicts("gt1", (range(3*n), range(9, numCliques-1)), cat="Binary")  # gt[i][k] = 1 ⇔ x[i][k]=1, x[i][k+1]=0 и до i все равны
for k in range(9, numCliques-1):
    model += u1[0][k] == 1
    for i in range(3*n-1):
        model += u1[i+1][k] <= u1[i][k]
        model += u1[i+1][k] <= eq1[i][k]
        model += u1[i+1][k] >= u1[i][k] - 1 + eq1[i][k]
for k in range(9, numCliques-1):
    for i in range(3*n):
        model += gt1[i][k] <= u1[i][k]
        model += gt1[i][k] <= vert[i][k]
        model += gt1[i][k] <= 1 - vert[i][k+1]
        model += gt1[i][k] >= u1[i][k] + vert[i][k] - vert[i][k+1] - 1
for k in range(9, numCliques-1):
    model += lpSum(gt1[i][k] for i in range(3*n)) == 1



for i in range(3 * n):
    for j in range(i + 1, 3 * n):
        if i // 3 != j // 3:
            model += lpSum(edges[i][j][k] for k in range(numCliques)) >= 1


print(pulp.listSolvers(onlyAvailable=True))
model.solve(pulp.getSolver('SCIP_PY'))

if model.status == 1:
    print("Solution found:")
    # for k in range(numCliques):
    #     if y[k].varValue == 1:
    #         print(f"Clique {k + 1} is chosen.")
    # for i in vertices:
    #     for j in vertices:
    #         for k in range(numCliques):
    #             if i < j and edges[i][j][k].varValue == 1:
    #                 print(f"Edge {i}-{j} is covered by Clique {k + 1}.")
    # for k in range(numCliques):
    #     print(edges[0][7][k].varValue)
                # if vert[i][k].varValue == 1 and vert[j][k].varValue == 1 and i < j:
                #     print(f"Edge {i}-{j} is covered by Clique {k + 1}.")



    print(numCliques)
    check_solution(vert)

    vert_solution = np.zeros((3 * n, numCliques))
    for v_idx in range(3 * n):
        for k_idx in range(numCliques):
            if vert[v_idx][k_idx].varValue is not None:
                vert_solution[v_idx, k_idx] = vert[v_idx][k_idx].varValue

    plt.figure(figsize=(max(10, numCliques * 0.6), max(8, 3 * n * 0.4)))
    plt.imshow(vert_solution, aspect='auto', cmap='Greys', interpolation='none')
    plt.title(f"Матрица выбора вершин для графа размера {3*n}", fontsize=14)
    plt.xlabel("Индекс клики (k)", fontsize=12)
    plt.ylabel("Индекс вершины (v)", fontsize=12)

    for i_triplet in range(1, n):
        plt.axhline(i_triplet * 3 - 0.5, color='dodgerblue', linewidth=1.0, linestyle='--')

    plt.yticks(np.arange(0, 3 * n, 1), labels=[str(i) for i in range(3 * n)])
    plt.xticks(np.arange(0, numCliques, 1), labels=[str(i) for i in range(numCliques)])

    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, numCliques, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3 * n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    total_vertices = 3 * n
    G = nx.Graph()
    G.add_nodes_from(range(3*n))

    edges = []
    for i in range(3*n):
        for j in range(i + 1, 3*n):
            if i // 3 != j // 3:
                edges.append((i, j))
    G.add_edges_from(edges)


    plt.figure(figsize=(16, 16))
    shells_for_layout = []
    for i_triplet in range(n):
        shells_for_layout.append([node for node in G.nodes() if node // 3 == i_triplet])

    pos_for_draw = nx.shell_layout(G, shells_for_layout)
    palette_for_nodes = plt.colormaps.get_cmap('viridis')
    node_colors= [palette_for_nodes(
            (node//3) / float(n - 1))
                                     for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos_for_draw, node_color=node_colors, node_size=350, alpha=0.9)
    nx.draw_networkx_labels(G, pos_for_draw, font_size=7)


    nx.draw_networkx_edges(G, pos_for_draw,
                               edgelist=edges,
                               edge_color='lightgray',
                               width=0.6)


    clique = 0
    active_nodes_for_k = [v for v in range(3*n) if vert[v][clique].varValue == 1]

    edges_in_k_clique = list(combinations(active_nodes_for_k, 2))

    nx.draw_networkx_edges(G, pos_for_draw,
                            edgelist=edges_in_k_clique,
                            edge_color='crimson',
                            width=2.5,
                            alpha=0.8)


    nx.draw_networkx_nodes(G, pos_for_draw,
                        nodelist=active_nodes_for_k,
                        node_color='crimson',
                        edgecolors='black', linewidths=1.0,
                        node_size=400)

    title_str = f"Граф на {3*n} вершинах с выделенными рёбрами клики k={clique}"

    plt.title(title_str, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()






else:
    print("No solution found.")










