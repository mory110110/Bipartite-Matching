#!/usr/bin/env python3.7


import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import itertools
import time

m = gp.Model("wobm")

dic= {}


""" Reads data from the specfied file and writes the graph tensor into multu dict of the following form:
    combinations, ms= gp.multidict({                                                                                 ('u1','v1'):10,
        ('u1','v2'):13,
        ('u2','v1'):9,
        ('u2','v2'):3
     })
"""

def get_data():
    E = np.random.randint(
        50, size=(50, 50)
    )  # for testing, size of (U,V), weigths vary from 1 to n-1
    u_size = E.shape[0]
    v_size = E.shape[1]

    # start = time.time()
    for u, v in itertools.product(range(u_size), range(v_size)):
        dic[(u, v)] = E[u][v]

    combinations, dic = gp.multidict(dic)
    print(type(combinations))
    print(type(dic))
    # end = time.time()

def run_IP():
    try:
        # add variable
        x = m.addVars(combinations, name="(u,v) pairs")

        # set constraints
        c1 = m.addConstrs((x.sum("*", v) <= 1 for v in range(v_size)), "V")
        c2 = m.addConstrs((x.sum(u, "*") <= 1 for u in range(u_size)), "U")

        # set the objective
        m.setObjective(x.prod(dic), GRB.MAXIMIZE)
        m.optimize()

        matched = 0
        for v in m.getVars():
            if abs(v.x) > 1e-6:
                matched += v.x

        print("total nodes matched: ", matched)
        print("total matching score: ", m.objVal)
        # print("time to generate the dictionary: ", end-start)

        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))
        except AttributeError:
            print("Encountered an attribute error")


if __name__ = "__main__":
    parser = argpasrse.ArgumentParser()

    parser.add_argument(
            "--dir",
            type=str,
            default="~/CORL/data/datset/train/graphs",
            help="which directory to read the data from and run IP solver on",
    )
    parser.add_argument(
            "--file",
            type=str,
            default="~/CORL/data/datset/train/graphs/0.pt",
            help="which file (graph) to run the IP solver on",
    )

get_data()
