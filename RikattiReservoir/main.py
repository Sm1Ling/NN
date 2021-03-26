import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
from threading import Thread
from threading import Lock
from ReSyst import *

M = 1200  # как долго высчитывать ошибку
S = 1000  # как долго запоминать вес
STEP = 0.04  # шаг подсчета времени
LEARNING_RATE = 0.05
NUM_OF_OSCILLATORS = 1

r_system = ReSyst(S, M, LEARNING_RATE, NUM_OF_OSCILLATORS)

lock = Lock()

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

plt.subplots_adjust(left=0.215,
                    bottom=0.05,
                    right=0.98,
                    top=0.93,
                    hspace=0.15)

texts = []
lines = []
oscillators_x = []

xdata, ydata, a_t, y_t = [], [], [], []

b_t = 1
c_t = 0.3


# Rikatti Equation  y'(t) = a(t) + b(t) * y(t) + c(t) * (y(t))^2
# y(t) = 5 * sin(sqr(x) + cos(x))
# y'(t) = 5 * cos( sqr(x) + cos(x)) / (2 * sqr(x)) - cos(sqr(x) + cos(x)) * sin(x)
# a(t) = 5 * cos(sqr(x) + cos(x)) * (1 / (2 * sqr(x)) - sin(x)) - 5 * b(t) * sin(sqr(x) + cos(x)) -
# - 5 * c(t) * (sin(sqr(x) + cos(x)))^2


def initializator():
    for i in range(17):
        texts.append(axes[0].text(-0.25, 0.95 - i * 0.05, '', transform=axes[0].transAxes))

    lines.append(axes[0].plot([], [], lw=2, color='r', label='My a(t)')[0])
    lines.append(axes[0].plot([], [], lw=2, color='g', label='y(t)')[0])
    lines.append(axes[0].plot([], [], lw=2, color='b', label='a(t)')[0])
    for i in range(NUM_OF_OSCILLATORS):
        lines.append(axes[1].plot([], [], lw=2, label=("x" + str(i + 1)))[0])

    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

    with open("LOG.csv", "w", newline="") as file:
        csv.writer(file, delimiter=';').writerows(
            [["Error", "Wx", "x", "y", "z", "dx", "dy", "dz",
              "dx\'(t)/dWx", "dy\'(t)/dWx", "dz\'(t)/dWx",
              "f", "Denominator", "Average", "dx(t)/dWx", "dy(t)/dWx", "dz(t)/dWx"]]
        )

    main()


def init():
    lines[0].set_data([], [])
    lines[1].set_data([], [])
    lines[2].set_data([], [])
    for i in range(NUM_OF_OSCILLATORS):
        lines[3+i].set_data([], [])
    return lines


def main():
    y_t.append(5*math.sin(math.sqrt(0.1) + math.cos(0.1)))
    a_t.append(5*math.cos(math.sqrt(0.1) + math.cos(0.1)) * (1/(2*math.sqrt(0.1))) - math.sin(0.1)
               - b_t * y_t[-1] - c_t * (y_t[-1]) ** 2)

    inp = [y_t[-1] for i in range(NUM_OF_OSCILLATORS)]
    out = [a_t[-1] for i in range(NUM_OF_OSCILLATORS)]

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
    t = 0.1

    while (True):

        t = (t + STEP)
        sqr_t = math.sqrt(t)

        y_t.append(5*math.sin(sqr_t + math.cos(t)))
        a_t.append(5*math.cos(sqr_t + math.cos(t)) * (1 / (2 * sqr_t) - math.sin(t)) - b_t * y_t[-1] - c_t * ((y_t[-1])**2))

        inp = [y_t[-1] for i in range(NUM_OF_OSCILLATORS)]
        out = [a_t[-1] for i in range(NUM_OF_OSCILLATORS)]

        lock.acquire()

        r_system.get_new_element(inp, out)

        xdata.append(t)

        av_out = 0
        for i in range(NUM_OF_OSCILLATORS):
            av_out += r_system.output[i]
            oscillators_x[i].append(r_system.oscillators[i].x[0])
        av_out = av_out/NUM_OF_OSCILLATORS

        ydata.append(av_out)

        if len(xdata) > 400:
            xdata.pop(0)
            ydata.pop(0)
            y_t.pop(0)
            a_t.pop(0)
            for i in range(NUM_OF_OSCILLATORS):
                oscillators_x[i].pop(0)

        lock.release()
        time.sleep(0.01)


def animate(i):
    lock.acquire()

    lines[0].set_data(xdata, ydata)
    lines[1].set_data(xdata, y_t)
    lines[2].set_data(xdata, a_t)
    for i in range(NUM_OF_OSCILLATORS):
        lines[3 + i].set_data(xdata, oscillators_x[i])

    if r_system.Error < 20:
        texts[0].set_text('error = %.3f' % r_system.Error)
        texts[1].set_text('wx = %.3f' % r_system.oscillators[0].wx)
        texts[2].set_text('dx = %.3f' % r_system.oscillators[0].dx[0])
        texts[3].set_text('dy= %.3f' % r_system.oscillators[0].dy)
        texts[4].set_text('dz = %.3f' % r_system.oscillators[0].dz)
        texts[5].set_text('dx\'(t)/dWx = %.3f' % r_system.oscillators[0].dxdwx[0])
        texts[6].set_text('dy\'(t)/dWx = %.3f' % r_system.oscillators[0].dydwx)
        texts[7].set_text('dz\'(t)/dWx = %.3f' % r_system.oscillators[0].dzdwx)
        texts[8].set_text('f = %.3f' % r_system.oscillators[0].f[0])
        texts[9].set_text('denominator = %.3f' % r_system.sqr_v)
        texts[10].set_text('average = %.3f' % r_system.av_x[0])
        texts[11].set_text('dx(t)/dWx = %.3f' % r_system.oscillators[0].dAdwx[0][0])
        texts[12].set_text('dy(t)/dWx = %.3f' % r_system.oscillators[0].dAdwx[1][0])
        texts[13].set_text('dz(t)/dWx = %.3f' % r_system.oscillators[0].dAdwx[2][0])
        texts[14].set_text('x = %.3f' % r_system.oscillators[0].x[0])
        texts[15].set_text('y = %.3f' % r_system.oscillators[0].y[0])
        texts[16].set_text('z = %.3f' % r_system.oscillators[0].z[0])

    with open("LOG.csv", "a", newline="") as file:
        csv.writer(file, delimiter=';').writerow(
            [r_system.Error, r_system.oscillators[0].wx, r_system.oscillators[0].x[0], r_system.oscillators[0].y[0],
             r_system.oscillators[0].z[0], r_system.oscillators[0].dx[0], r_system.oscillators[0].dy,
             r_system.oscillators[0].dz,
             r_system.oscillators[0].dxdwx[0], r_system.oscillators[0].dydwx, r_system.oscillators[0].dzdwx,
             r_system.oscillators[0].f[0], r_system.sqr_v, r_system.av_x[0], r_system.oscillators[0].dAdwx[0][0],
             r_system.oscillators[0].dAdwx[1][0], r_system.oscillators[0].dAdwx[2][0]]
        )

    axes[0].set_xlim(xdata[0], xdata[-1])
    axes[1].set_xlim(xdata[0], xdata[-1])
    axes[0].set_ylim(-20, 20)
    axes[1].set_ylim(-20, 20)

    lock.release()
    return lines


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initializator()
