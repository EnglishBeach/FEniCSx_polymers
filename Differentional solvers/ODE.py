import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# dy/dt=F
def u(y):
    return y * 20

def F(y, t, k):
    dydt = -k * u(y)
    return dydt

def main():
    y0 = 5
    t = np.linspace(0, 10)
    y = odeint(F, y0, t, args=(0.2, ))
    plt.plot(t, y)
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.show()


main()