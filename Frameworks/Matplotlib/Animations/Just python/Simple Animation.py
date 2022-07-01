import matplotlib.pyplot as plt
import numpy as np


# Bad way
def Clear_view():
    x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    import time
    plt.ion()
    for space in np.arange(0, 10, 0.1):
        y = np.cos(x + space)

        plt.clf()
        plt.plot(x, y)

        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.02)
    plt.ioff()
    plt.show()


def Object_view():
    import time

    plt.ion()
    fig, axes = plt.subplots()

    x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    y = x

    line, = axes.plot(x, y)  #hyper to plot axes

    for space in np.arange(0, 10, 0.1):
        y = np.cos(x + space)

        line.set_ydata(y)

        plt.draw()
        plt.gcf().canvas.flush_events()

        time.sleep(0.02)
    plt.ioff()
    plt.show()


def Good_view():
    from matplotlib.animation import FuncAnimation as fa

    fig, axes = plt.subplots()
    x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)

    y = np.cos(x)
    line, = axes.plot(x, y)  #line - hyper to plot axes

    def f(param, line, x):  #Important: 3 pframs
        line.set_ydata(np.cos(x + param))
        return [line]

    phase_frame = np.arange(-np.pi, 0, 0.1)

    animfunc = fa(
        fig,
        func=f,
        frames=phase_frame,  # Parametr
        fargs=(line, x),
        interval=30,
        blit=True)
    plt.show()

Good_view()