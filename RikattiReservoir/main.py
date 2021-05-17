import csv
import matplotlib;

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from threading import Thread
from threading import Lock

from FunctionClasses import RikattiFunc, ThermalFunc
from ReSyst import *

M = 20  # как долго высчитывать ошибку
S = 10  # как долго запоминать вес
STEP = 0.05  # шаг подсчета времени
LEARNING_RATE = 0.05
NUM_OF_OSCILLATORS = 1
GRAPHIC_SIZE = 400

r_system = ReSyst(S, M, LEARNING_RATE, NUM_OF_OSCILLATORS)

lock = Lock()

fig, axes = plt.subplots(3, 1, figsize=(10, 7))

plt.subplots_adjust(left=0.215,
                    bottom=0.05,
                    right=0.98,
                    top=0.93,
                    hspace=0.15)

texts = []
lines = []
oscillators_x = []

xdata, ydata, a_t, y_t = [], [], [], []

functions = [RikattiFunc(1, 0.3)]


def initializator():
    for i in range(3):
        texts.append(axes[0].text(-0.25, 0.95 - i * 0.1, '', transform=axes[0].transAxes))
    texts.append(axes[0].text(-0.25, 0, '', transform=axes[1].transAxes))

    lines.append(axes[0].plot([], [], lw=2, color='r', label='My a(t)')[0])
    lines.append(axes[0].plot([], [], lw=2, color='g', label='t')[0])
    lines.append(axes[0].plot([], [], lw=2, color='b', label='a(t)')[0])
    for i in range(NUM_OF_OSCILLATORS):
        lines.append(axes[1].plot([], [], lw=2, label=("x" + str(i + 1)))[0])

    axes[2].bar([], [])

    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

    main()


def init():
    lines[0].set_data([], [])
    lines[1].set_data([], [])
    lines[2].set_data([], [])
    for i in range(NUM_OF_OSCILLATORS):
        lines[3 + i].set_data([], [])
    return lines


def main():
    y_t.append(functions[0].input_func(0.1))
    a_t.append(functions[0].output_func())


    inp = [y_t[-1] for _ in range(NUM_OF_OSCILLATORS)]
    out = [a_t[-1] for _ in range(NUM_OF_OSCILLATORS)]

    r_system.get_new_element(inp, out)  # 0.01

    xdata.append(0.1)
    ydata.append(r_system.output[0])  # по первому осциллятору

    for i in range(NUM_OF_OSCILLATORS):
        oscillators_x.append([0.1])

    thread1 = Thread(target=parallelstart, args=())
    thread1.start()

    anim = animation.FuncAnimation(fig, animate, frames=50, init_func=init, interval=60)
    plt.show()


def parallelstart():
    t = 0
    f_index = 0
    while True:
        lock.acquire()

        t = (t + STEP)

        y_t.append(functions[f_index].input_func(t))
        a_t.append(functions[f_index].output_func())

        inp = [y_t[-1] for _ in range(NUM_OF_OSCILLATORS)]
        out = [a_t[-1] for _ in range(NUM_OF_OSCILLATORS)]

        r_system.get_new_element(inp, out)

        xdata.append(t)

        av_out = 0
        for i in range(NUM_OF_OSCILLATORS):
            av_out += r_system.output[i]
            oscillators_x[i].append(r_system.oscillators[i].x.get_last())
        av_out = av_out / NUM_OF_OSCILLATORS

        ydata.append(av_out)

        if len(xdata) > GRAPHIC_SIZE:
            xdata.pop(0)
            ydata.pop(0)
            y_t.pop(0)
            a_t.pop(0)
            for i in range(NUM_OF_OSCILLATORS):
                oscillators_x[i].pop(0)

        lock.release()
        time.sleep(0.02)


def animate(i):
    lock.acquire()

    lines[0].set_data(xdata, ydata)
    lines[1].set_data(xdata, y_t)
    lines[2].set_data(xdata, a_t)
    for j in range(NUM_OF_OSCILLATORS):
        lines[3 + j].set_data(xdata, oscillators_x[j])
    for bar in axes[2].containers:
        bar.remove()
    axes[2].bar(range(len(r_system.oscillators[0].w)), r_system.oscillators[0].w, color="black")

    if r_system.Error < 100:
        texts[0].set_text('Error = %.3f' % r_system.Error)
        texts[1].set_text('Denominator = %.3f' % r_system.sqr_v)
        texts[2].set_text('Average X = %.3f' % r_system.av_x.get_last())
        texts[3].set_text(str(r_system.oscillators[0]))

    axes[0].set_xlim(xdata[0], xdata[-1])
    axes[1].set_xlim(xdata[0], xdata[-1])
    axes[0].set_ylim(-20, 10)
    axes[1].set_ylim(-10, 10)
    axes[2].set_ylim(-2,2)

    lock.release()
    return lines




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initializator()
