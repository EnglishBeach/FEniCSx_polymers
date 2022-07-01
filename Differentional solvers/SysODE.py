import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# System model:
def vessel(F, t, qout, qin, Cf, Tf):
    # Parameters
    V = F[0]
    C = F[1]
    T = F[2]
    r=0

    # Mass balance ro=Const
    dVdt = qin - qout

    # Concentration balance
    # d(V*C)/s]dt = C*dV/dt + V*dC/dt
    dCdt = (qin * Cf - qout * C) / V - r - (C * dVdt) / V

    # Energy balance
    # d(T*V)/dt = T*dV/dt + V*dT/dt
    dTdt = (qin * Tf - qout * T) / V - (T * dVdt) / V
    return [dVdt, dCdt, dTdt]


def main():
    V0 = 1.0
    C0 = 0.0
    T0 = 350.0
    F0 = [V0, C0, T0]

    # Time
    t = np.linspace(0, 10, 100)

    # Reaction
    r = np.zeros(len(t))

    # Flows
    qin = np.ones(len(t)) * 5.2
    qin[50:] = 5.1
    qout = np.ones(len(t)) * 5.0

    # Concentration
    Cf = np.ones(len(t)) * 1.0
    Cf[30:] = 0.5

    # Temperature
    Tf = np.ones(len(t)) * 300.0
    Tf[70:] = 325.0

    # Storage
    V = np.ones(len(t)) * V0
    C = np.ones(len(t)) * C0
    T = np.ones(len(t)) * T0

    # Simulate
    for it in range(len(t)-1):
        inputs = (qout[it], qin[it], Cf[it], Tf[it])
        ts = [t[it], t[it + 1]]
        F = odeint(vessel, F0, ts, args=inputs)

        # Restore
        V[it + 1] = F[-1][0]
        C[it + 1] = F[-1][1]
        T[it + 1] = F[-1][2]
        F0 = F[-1]

    # Plot the inputs and results
    plt.figure()

    plt.subplot(3, 2, 1)
    plt.plot(t, qin, 'b--', linewidth=3)
    plt.plot(t, qout, 'b:', linewidth=3)
    plt.ylabel('Flow Rates (L/min)')
    plt.legend(['Inlet', 'Outlet'], loc='best')

    plt.subplot(3, 2, 3)
    plt.plot(t, Cf, 'r--', linewidth=3)
    plt.ylabel('Cf (mol/L)')
    plt.legend(['Feed Concentration'], loc='best')

    plt.subplot(3, 2, 5)
    plt.plot(t, Tf, 'k--', linewidth=3)
    plt.ylabel('Tf (K)')
    plt.legend(['Feed Temperature'], loc='best')
    plt.xlabel('Time (min)')

    plt.subplot(3, 2, 2)
    plt.plot(t, V, 'b-', linewidth=3)
    plt.ylabel('Volume (L)')
    plt.legend(['Volume'], loc='best')

    plt.subplot(3, 2, 4)
    plt.plot(t, C, 'r-', linewidth=3)
    plt.ylabel('C (mol/L)')
    plt.legend(['Concentration'], loc='best')

    plt.subplot(3, 2, 6)
    plt.plot(t, T, 'k-', linewidth=3)
    plt.ylabel('T (K)')
    plt.legend(['Temperature'], loc='best')
    plt.xlabel('Time (min)')

    plt.show()


main()