#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/14 21:27
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : PPS.py
# @Statement : Push and pull search (PPS) is a constrained multi-objective evolutionary algorithm (CMOEA)
# @Reference : Fan Z, Li W, Cai X, et al. Push and pull search for solving constrained multi-objective optimization problems[J]. Swarm and Evolutionary Computation, 2019, 44: 665-679.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform


def cal_obj(x):
    # LIR-CMOP6
    J1 = [2 * i + 1 for i in range(1, 15)]
    J2 = [2 * i for i in range(1, 16)]
    g1 = sum([(x[i - 1] - np.sin(0.5 * i * np.pi * x[0] / 30)) ** 2 for i in J1])
    g2 = sum([(x[i - 1] - np.cos(0.5 * i * np.pi * x[0] / 30)) ** 2 for i in J2])
    f1 = x[0] + 10 * g1 + 0.7057
    f2 = 1 - x[0] ** 2 + 10 * g2 + 0.7057
    c1 = ((f1 - 1.8) * np.cos(-0.25 * np.pi) - (f2 - 1.8) * np.sin(-0.25 * np.pi)) ** 2 / 4 + (
                (f1 - 1.8) * np.sin(-0.25 * np.pi) + (f2 - 1.8) * np.cos(-0.25 * np.pi)) ** 2 / 64
    c2 = ((f1 - 2.8) * np.cos(-0.25 * np.pi) - (f2 - 2.8) * np.sin(-0.25 * np.pi)) ** 2 / 4 + (
                (f1 - 2.8) * np.sin(-0.25 * np.pi) + (f2 - 2.8) * np.cos(-0.25 * np.pi)) ** 2 / 64
    return [f1, f2], abs(max(0.1 - c1, 0)) + abs(max(0.1 - c2, 0))


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from a n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, dim):
    # calculate approximately npop uniformly distributed reference points on dim dimensions
    h1 = 0
    while combination(h1 + dim, dim - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + dim), dim - 1))) - np.arange(dim - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < dim:
        h2 = 0
        while combination(h1 + dim - 1, dim - 1) + combination(h2 + dim, dim - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + dim), dim - 1))) - np.arange(dim - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * dim)
            points = np.concatenate((points, temp_points), axis=0)
    points = np.where(points != 0, points, 1e-3)
    return points


def update_rk(ideal, nadir, t, last_gen):
    # update the maximum rate of change od ideal and nadir points
    delta = np.full(ideal.shape[1], 1e-6)
    rzk = np.max(np.abs(ideal[t] - ideal[t - last_gen]) / np.max((ideal[t - last_gen], delta), axis=0))
    rnk = np.max(np.abs(nadir[t] - nadir[t - last_gen]) / np.max((nadir[t - last_gen], delta), axis=0))
    return max(rnk, rzk)


def update_epsilon(tau, epsilon_k, epsilon_0, rf, alpha, t, Tc, cp):
    # update epsilon
    if rf < alpha:
        return (1 - tau) * epsilon_k
    else:
        return epsilon_0 * ((1 - t / Tc) ** cp)


def mutation(individual, lb, ub, dim, pm, eta_m):
    # polynomial mutation
    if np.random.random() < pm:
        site = np.random.random(dim) < 1 / dim
        mu = np.random.random(dim)
        delta1 = (individual - lb) / (ub - lb)
        delta2 = (ub - individual) / (ub - lb)
        temp = np.logical_and(site, mu <= 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
        temp = np.logical_and(site, mu > 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
        individual = np.where(((individual >= lb) & (individual <= ub)), individual, np.random.uniform(lb, ub))
    return individual


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def nd_cd_sort(pop, objs, npop):
    # sort the population according to the Pareto rank and crowding distance
    pfs, rank = nd_sort(objs)
    cd = crowding_distance(objs, pfs)
    if len(pfs[1]) <= npop:
        return pop[pfs[1]], objs[pfs[1]]
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pop = np.zeros((npop, pop.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    for i in range(npop):
        next_pop[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
    return next_pop, next_objs


def main(npop, iter, lb, ub, T=30, delta=0.9, nr=2, tau=0.1, alpha=0.95, Tc=0.8, cp=2, last_gen=20, CR=1, F=0.5, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param T: neighborhood size (default = 30)
    :param delta: the probability of selecting individuals in the neighborhood (default = 0.9)
    :param nr: the maximal number of solutions replaced by a child (default = 2)
    :param tau: control the scale factor multiplied by the maximum overall constraint violation (default = 0.1)
    :param alpha: control the searching preference between the feasible and infeasible regions (default = 0.95)
    :param Tc: control generation (default = 0.8 * iter)
    :param cp: control the speed of reducing relaxation of constraints (default = 2)
    :param last_gen: the last generation (default = 20)
    :param CR: crossover rate (default = 1)
    :param F: mutation scalar number (default = 0.5)
    :param eta_m: spread factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    Tc *= iter
    nvar = len(lb)  # the dimension of decision space
    nobj = len(cal_obj((lb + ub) / 2))  # the dimension of objective space
    ideal = np.zeros((iter, nobj))  # the ideal points
    nadir = np.zeros((iter, nobj))  # the nadir points
    V = reference_points(npop, nobj)  # weight vectors
    sigma = squareform(pdist(V, metric='euclidean'), force='no', checks=True)  # distances between weight vectors
    B = np.argsort(sigma)[:, : T]  # the T closet weight vectors
    npop = V.shape[0]
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = np.array([cal_obj(x)[0] for x in pop])  # objectives
    phi = np.array([cal_obj(x)[1] for x in pop])  # constraint violations
    phi_max = np.max(phi)  # the maximum overall constraint violation found so far
    epsilon_k = epsilon_0 = 0
    push_stage = True  # denotes the search is in the push stage
    rk = 1  # the max rate of change between the ideal and nadir points during the last last_gen generations
    rf = np.where(phi == 0)[0].shape[0] / npop  # the ratio of feasible solutions to all solutions
    z = np.min(objs, axis=0)  # ideal point
    NS, NS_objs = nd_cd_sort(pop, objs, npop)  # feasible non-dominated solutions

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Update ideal and nadir points
        ideal[t] = z.copy()
        nadir[t] = np.max(objs, axis=0)

        # Step 2.2. Update rk
        if t >= last_gen:
            rk = update_rk(ideal, nadir, t, last_gen)

        # Step 2.3. Update epsilon
        if t < Tc:
            if rk <= 1e-3 and push_stage:
                push_stage = False
                epsilon_0 = epsilon_k = phi_max
            if not push_stage:
                epsilon_k = update_epsilon(tau, epsilon_k, epsilon_0, rf, alpha, t, Tc, cp)
        else:
            epsilon_k = 0

        # Step 2.4. DE operator + polynomial mutation
        for i in range(npop):
            if np.random.random() < delta:
                P = np.random.permutation(B[i])
            else:
                P = np.random.permutation(npop)
            if np.random.random() < CR:
                off = pop[i] + F * (pop[P[0]] - pop[P[1]])
                off = np.where(((off >= lb) & (off <= ub)), off, np.random.uniform(lb, ub))
            else:
                off = pop[i].copy()
            off = mutation(off, lb, ub, nvar, 1, eta_m)
            off_obj, off_phi = cal_obj(off)
            z = np.min((z, off_obj), axis=0)
            phi_max = max(phi_max, off_phi)
            c = 0  # update counter
            if push_stage:  # push stage
                for j in P:
                    if c == nr:
                        break
                    if np.max(V[j] * np.abs(off_obj - z)) < np.max(V[j] * np.abs(objs[j] - z)):
                        c += 1
                        pop[j] = off
                        objs[j] = off_obj
                        phi[j] = off_phi
            else:  # pull stage
                for j in P:
                    if c == nr:
                        break
                    flag = False
                    if (phi[j] <= epsilon_k and off_phi <= epsilon_k) or phi[j] == off_phi:
                        if np.max(V[j] * np.abs(off_obj - z)) < np.max(V[j] * np.abs(objs[j] - z)):
                            flag = True
                    elif off_phi < phi[j]:
                        flag = True
                    if flag:
                        c += 1
                        pop[j] = off
                        objs[j] = off_obj
                        phi[j] = off_phi

        # Step 2.7. Update rk and NS
        ind = np.where(phi == 0)[0]
        rf = ind.shape[0] / npop
        NS, NS_objs = nd_cd_sort(np.concatenate((NS, pop[ind]), axis=0), np.concatenate((NS_objs, objs[ind]), axis=0), npop)

    # Step 3. Sort the results
    plt.figure()
    x = [o[0] for o in NS_objs]
    y = [o[1] for o in NS_objs]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of LIR-CMOP6')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(300, 1000, np.array([0] * 30), np.array([1] * 30))
