import mph
import numpy as np
from  matplotlib import pyplot as plt

# load mph model
client = mph.start(cores=4)
client.clear()
java = client.java
py_model = client.load('fluid2d.mph')
model = py_model.java


if __name__ == "__main__":
    n = 100
    dy = 1/n
    Tw = 400
    L = '0.2[cm]'

    int_Tm = "abs(v)*T"
    int_v = "abs(v)"

    y = np.arange(0, n)*dy
    Nu = []
    Tm = []

    for i in range(n):

        yi = i*dy
        tag_cutline = "cln3"
        tag_int = 'int3'
        tag_cutPoint = "cpt1"
        tag_evalPoint = "pev1"

        # create a Cut Line
        model.result().dataset().create(tag_cutline, "CutLine2D")
        model.result().dataset(tag_cutline).setIndex("genpoints", 0.0, 0, 0)
        model.result().dataset(tag_cutline).setIndex("genpoints", 0.2, 1, 0)
        model.result().dataset(tag_cutline).setIndex("genpoints", yi, 1, 1)
        model.result().dataset(tag_cutline).setIndex("genpoints", yi, 0, 1)
        # create an integral operation along the Cut Line
        model.result().numerical().create(tag_int, "IntLine")
        model.result().numerical(tag_int).set("intsurface", True)
        model.result().numerical(tag_int).set("data", tag_cutline)
        model.result().numerical(tag_int).setIndex("expr", int_Tm, 0)

        # calculating integral and data extraction
        data1 = model.result().numerical(tag_int).getReal()
        data1 = np.array(data1).flatten()
        # calculating the integral and data extraction
        model.result().numerical(tag_int).setIndex("expr", int_v, 0)
        data2 = model.result().numerical(tag_int).getReal()
        data2 = np.array(data2).flatten()
        print(f"Calculating average Temperature at y={yi:.3f}cm: {data1/data2}")
        Tm.append(data1/data2)

        # create Cut Point to evaluate Tx(x=0)
        model.result().dataset().create(tag_cutPoint, "CutPoint1D")
        # set x position at x=0
        model.result().dataset(tag_cutPoint).set("data", tag_cutline)
        model.result().dataset(tag_cutPoint).set("pointx", 0.0)

        model.result().numerical().create(tag_evalPoint, "EvalPoint")
        model.result().numerical(tag_evalPoint).set("data", tag_cutPoint)
        model.result().numerical(tag_evalPoint).setIndex("expr", f'{L}*Tx', 0)

        Tx = model.result().numerical(tag_evalPoint).getReal()
        data3 = -np.array(Tx).flatten()/(Tw-Tm[-1])

        Nu.append(data3)

        # delete the Cut Line and integral operation along the Cut Line
        model.result().numerical().remove(tag_evalPoint)
        model.result().dataset().remove(tag_cutPoint)
        model.result().numerical().remove(tag_int)
        model.result().dataset().remove(tag_cutline)

    # plot
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(y, Nu, label='COMSOL')
    ax1.set_xlabel('y (cm)')
    ax1.set_ylabel(r'$Nu_{y}$')
    ax1.legend()
    plt.savefig('Nusselt number along y')

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(y, Tm, label='COMSOL')
    ax1.set_xlabel('y (cm)')
    ax1.set_ylabel(r'Average $T_{bulk}$')
    ax1.legend()
    plt.savefig('Average cross section Temperature along y')
