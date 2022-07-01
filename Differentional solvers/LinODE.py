import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(z, u, t):
    x1 = z[0]
    x2 = z[1]
    dx1dt = -x1**2 + np.sqrt(u)
    dx2dt = -4. * (x2 - 2.) + 1 / 8. * (u - 16.)
    dzdt = [dx1dt, dx2dt]
    return dzdt


def main():
    x0 = 2.
    u0 = 16.
    z0 = [x0, x0]

    final_time = 10
    steps_time = 10 * final_time + 1
    t = np.linspace(0, final_time, steps_time)

    u = np.ones(steps_time) * u0
    x1=np.ones_like(t)*x0
    x2=np.ones_like(t)*x0
    for i in range(1, steps_time):
        span_time = [t[i - 1], t[i]]
        z = odeint(model, z0, span_time, args=(u[i], ))
        z0 = z[1]
        x1[i] = z[1][0]
        x2[i] = z[1][1]

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, u, 'g-', linewidth=3, label='u(t) Doublet Test')
    plt.grid()
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(t, x1, 'b-', linewidth=3, label='x(t) Nonlinear')
    plt.plot(t, x2, 'r--', linewidth=3, label='x(t) Linear')
    plt.xlabel('time')
    plt.grid()
    plt.legend(loc='best')
    plt.show()


main()