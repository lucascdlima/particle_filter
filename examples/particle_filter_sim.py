# Script for executing Particle Filter Simulation
import numpy as np
from numpy import pi
import tests.context
from particle_filter import particlefilter as pf
import numpy.matlib

from particle_filter.guiparticlefilter import QtWidgets, ApplicationWindow
from particle_filter.particlefilter import particle_filter_simulation

import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from tests.animation_test import animation_test

if __name__ == "__main__":
   import sys

   fig1, ax1 = plt.subplots()
   x0 = np.array((0.0,0.0,0.0))
   odom_variance = [0.6, 0.4, 0.4, 0.4]
   landmark_variance = [0.2, 0.2, 0.1]
   M_particles= 500
   T = 20
   dt = 0.1
   particle_filter_simulation(x0, odom_variance, landmark_variance,M_particles, T, dt, "",ax1, plt)

   """ if len(sys.argv) > 1:
        if sys.argv[1] == "animate":
            animate = "animate"
        else:
            animate = ""
    else:
        animate = ""

    theta0 = 0
    x0 = np.array([0, 0, theta0], dtype=float)
    xreal = np.copy(x0)
    xodot_1 = np.copy(x0)

    T = 20
    dt = 0.1
    u = np.array([0.15, 20 * pi / 180], dtype=float)
    # alpha = [0.07 0.05 0.05 0.05]; %moving aprox 1 m, errors = aprox 0.1 m

    alpha = np.array([0.8, 0.6, 0.6, 0.6])  # variance of odometric readings
    varsen = np.array([0.15, 0.15, 0.1])  # variance of sensor landmark readings

    m = np.stack((np.concatenate((np.arange(5), np.arange(5))), np.concatenate((np.ones(5) * 2, np.ones(5) * 3)),
                  np.linspace(1, 10, 10)))  # map of environment (landmarks)

    M = 500
    mapxend = 8
    mapyend = 6

    xinit = -0.5 + (mapxend - (-0.5)) * np.random.rand(M)
    yinit = -0.5 + (mapyend - (-0.5)) * np.random.rand(M)
    thetainit = np.zeros((M))

    X0 = np.stack((xinit, yinit, thetainit, np.zeros(M)))

    t = np.arange(0, T, dt)

    u_vec_temp = np.repeat(np.array([[u[0], u[0]], [u[1], -u[1]]]), [round(5 / dt, 1), round(5 / dt, 1)], axis=1)

    u_vec = np.matlib.repmat(u_vec_temp, 1, round(t.size / 5 / 2))

    xodot1 = np.copy(x0)
    xold = np.copy(xodot1)

    fig1, ax1 = plt.subplots()
    ax1.set_xlim([-2, 5])
    ax1.set_ylim([-2, 5])

    col = ['r', 'g', 'b', 'c', 'm', 'k']

    ax1.scatter(m[0, :], m[1, :], c='w', s=120, marker='^', edgecolors='g')

    Xt = np.copy(X0)
    Xtold = np.copy(Xt)
    k = 1

    len = t.size

    xodo_salva = np.zeros((3, len))
    xreal_salva = np.zeros((3, len))

    start_time = time.time()
    Xt_part = []

    for i in range(0, len, 1):

        xodot1[0] = xodot_1[0] + u_vec[0, i] * dt * np.cos(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[1] = xodot_1[1] + u_vec[0, i] * dt * np.sin(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[2] = xodot_1[2] + u_vec[1, i] * dt

        ut = np.concatenate((xodot_1, xodot1))
        xreal = pf.sample_motion_model_odometry(ut, xreal, alpha)

        zt = np.stack((np.sqrt((m[0, :] - xreal[0]) ** 2 + (m[1, :] - xreal[1]) ** 2) + varsen[0] * np.random.randn(),
                       np.arctan2(m[1, :] - xreal[1], m[0, :] - xreal[0]) - xreal[2] + varsen[1] * np.random.randn(),
                       m[2, :]))

        Xt = pf.particle_filter_algorithm(Xtold, M, ut, m, zt, alpha, varsen)
        Xtold = np.copy(Xt)

        if t[i] % 4 == 0:
            ax1.scatter(Xt[0, :], Xt[1, :], 2, col[k], zorder=3)
            k = k + 1

        xodot_1 = np.copy(xodot1)
        xodo_salva[:, i] = np.copy(xodot1)
        xreal_salva[:, i] = np.copy(xreal)
        print("Percentual exec: %f " % (t[i] * 100 / T))

    dtime = time.time() - start_time
    print("execution time: %s seconds" % (dtime))

    ax1.plot(xodo_salva[0, :], xodo_salva[1, :], c='r')

    if animate == "animate":
        line, = ax1.plot([], [], lw=2, c='g')
        anim = animation.FuncAnimation(fig1, animation_test, fargs=(xreal_salva, ax1, line), frames=len, interval=20)
    else:
        ax1.scatter(xreal_salva[0, :], xreal_salva[1, :], s=10, c='w', edgecolors='g')
        ax1.plot(xreal_salva[0, :], xreal_salva[1, :], c='g')

    plt.show()"""
