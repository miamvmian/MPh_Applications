import mph
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import pandas as pd

import logging
import sys
#
#
# logger = logging.getLogger('mph')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# load mph model
client = mph.start(cores=4, version='6.2')
client.clear()
java = client.java
py_model = client.load('masterClass_2D.mph')
model = py_model.java
#
# res = py_model.evaluate(['ewfd.Ebr', 'r', 'z'])
# Ebr = res[0]
# r = res[1]
# z = res[2]

def comsol_action_rz1(x):
    model.param().set('r1', f'{x[0]}[nm]')
    model.param().set('r2', f'{x[1]}[nm]')
    model.param().set('r3', f'{x[0]}[nm]')
    # model.param().set('z1', f'{x[3]}[nm]')
    # model.param().set('z2', f'{x[4]}[nm]')

    # model.component("comp1").geom("geom1").feature("c2").set("pos", {"-10[nm]", "0"})
    model.component("comp1").geom("geom1").run("fin")
    model.component("comp1").mesh("mesh1").run()

    py_model.solve()
    res = py_model.evaluate('SCS / (pi*(75[nm])^2)')
    return res

def max_f1(x):
    res = comsol_action_rz1(x)
    # res = np.around(res, 3)
    print(np.around(x, 3))
    print(res)
    return -res

#
#
# def comsol_action_rz2(x):
#     # model.param().set('r1', f'{x[0]}[nm]')
#     # model.param().set('r2', f'{x[1]}[nm]')
#     model.param().set('r3', f'{x[0]}[nm]')
#     # model.param().set('z1', f'{x[3]}[nm]')
#     # model.param().set('z2', f'{x[4]}[nm]')
#
#     # model.component("comp1").geom("geom1").feature("c2").set("pos", {"-10[nm]", "0"})
#     model.component("comp1").geom("geom1").run("fin")
#     model.component("comp1").mesh("mesh1").run()
#
#     py_model.solve()
#     res = py_model.evaluate('SCS / (pi*(75[nm])^2)')
#     return res
#
#
# def max_f2(x):
#     res = comsol_action_rz2(x)
#     # res = np.around(res, 3)
#     print(np.around(x, 3))
#     print(res)
#     return -res


def comsol_action_rz3(x):
    # model.param().set('r1', f'{x[0]}[nm]')
    # model.param().set('r2', f'{x[1]}[nm]')
    # model.param().set('r3', f'{x[0]}[nm]')
    model.param().set('z1', f'{x[0]}[nm]')
    model.param().set('z2', f'{x[1]}[nm]')
    model.component("comp1").geom("geom1").run("fin")
    model.component("comp1").mesh("mesh1").run()

    py_model.solve()
    res = py_model.evaluate('SCS / (pi*(75[nm])^2)')
    return res

def max_f3(x):
    res = comsol_action_rz3(x)
    # res = np.around(res, 3)
    print(np.around(x, 3))
    print(res)
    return -res


def comsol_action_rz(x):
    model.param().set('r1', f'{x[0]}[nm]')
    model.param().set('r2', f'{x[1]}[nm]')
    model.param().set('r3', f'{x[0]}[nm]')
    model.param().set('z1', f'{x[2]}[nm]')
    model.param().set('z2', f'{x[3]}[nm]')
    model.component("comp1").geom("geom1").run("fin")
    model.component("comp1").mesh("mesh1").run()

    py_model.solve()
    res = py_model.evaluate('SCS / (pi*(75[nm])^2)')
    return res

def max_f(x):
    res = comsol_action_rz(x)
    # res = np.around(res, 3)
    print(np.around(x, 3))
    print(res)
    return -res


def min_f(x):
    res = comsol_action_rz(x)
    print(np.around(x, 3))
    print(res)
    return res


if __name__ == "__main__":
    py_model.parameter('r1', '75[nm]')
    py_model.parameter('r2', '75[nm]')
    py_model.parameter('r3', '75[nm]')

    py_model.parameter('z1', '275[nm]')
    py_model.parameter('z2', '-275[nm]')
    #
    df = pd.read_csv('lda_radius.csv')
    lda_l = df['lda'].values
    r_l = df['r'].values

    r1 = 71.9
    r2 = 94.5
    r3 = 71.9
    z1 = 315.2
    z2 = -314.5

    # for ldai in range(600,370,-30):
    ldai = 600
    print(ldai)
    model.param().set('lda', f'{ldai}[nm]')
    bounds=[[50,120], [50,120]]
    results = optimize.minimize(max_f1, x0=np.array([r1, r2]),
                                bounds=bounds, tol=1e-1, method='Nelder-Mead', )
    print(f'local maximum parameters are: {results}')

    bounds = ([150, 420], [-420, -150])
    results3 = optimize.minimize(max_f3, x0=np.array([300, -300]), bounds=bounds,
                                 tol=1e-1, method='Nelder-Mead', )
    print(f'local maximum parameters are: {results3}')

    r1, r2, z1, z2 = py_model.evaluate(['r1', 'r2', 'z1', 'z2'])
    bounds = ([r1-ldai/4/2, r1+ldai/4/2],
              [r2-ldai/4/2, r2+ldai/4/2],
              [z1-50, z1+50],[z2-50, z2+50])
    results = optimize.minimize(max_f, x0=np.array([r1,r2,z1,z2]), bounds=bounds,
                                 tol=1e-2, method='Nelder-Mead')

    print(f'local minimum parameters are: {results}')

    r1, r2, r3, z1, z2 = py_model.evaluate(['r1', 'r2', 'r3', 'z1', 'z2'])
    print(f"parameters: r1={r1}")
    print(f"parameters: r2={r2}")
    print(f"parameters: r3={r3}")
    print(f"parameters: z1={z1}")
    print(f"parameters: z2={z2}")
    print(f"2* SCS / (pi*(75[nm])^2) = {2*py_model.evaluate('SCS / (pi*(75[nm])^2)')}")
