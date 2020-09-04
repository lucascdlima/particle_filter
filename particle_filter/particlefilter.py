# Particle filter module
import numpy as np
from numpy import pi
from scipy.stats import norm
import time


def motion_model_odometry(xt, ut, xt_1, odom_std):
    """Calculates a density probability function of a particle in position xt given the known
    odometry measure ut and last position xt_1. In other words, computes the probability p(xt/ut,xt_1).
      xt: array[3] current unknown position
      odom_std = array[4] of standard deviations in odometry measures
      ut: array[6] of odometry positions in instants (t-1) and (t)
      xt_1: array[3] of known position in instant t-1
      """

    a1 = odom_std[0]
    a2 = odom_std[1]
    a3 = odom_std[2]
    a4 = odom_std[3]

    x_bar = ut[0]
    y_bar = ut[1]
    theta_bar = ut[2]
    x_bar_l = ut[3]
    y_bar_l = ut[4]
    theta_bar_l = ut[5]

    x = xt_1[0]
    y = xt_1[1]
    theta = xt_1[2]

    x_l = xt[0]
    y_l = xt[1]
    theta_l = xt[2]

    #Calculates translations and rotations given odometry measures
    dtrans = np.sqrt((x_bar_l - x_bar) ** 2 + (y_bar_l - y_bar) ** 2)
    drot1 = np.arctan2(y_bar_l - y_bar, x_bar_l - x_bar) - theta_bar
    drot2 = theta_bar_l - theta_bar - drot1

    dtrans_hat = np.sqrt((x_l - x) ** 2 + (y_l - y) ** 2)
    drot1_hat = np.arctan2(y_l - y, x_l - x) - theta
    drot2_hat = theta_l - theta - drot1_hat

    diff_rot1 = angle_abs_pi(drot1,drot1_hat)
    diff_rot2 = angle_abs_pi(drot2, drot2_hat)

    prob_1 = norm.pdf(diff_rot1, loc=0, scale=a1 * abs(drot1_hat) + a2 * abs(dtrans_hat))
    prob_2 = norm.pdf(dtrans - dtrans_hat, loc=0, scale=a3 * abs(dtrans_hat) + a4 * (abs(drot1_hat) + abs(drot2_hat)))
    prob_3 = norm.pdf(diff_rot2, loc=0, scale=a1 * abs(drot2_hat) + a2 * abs(dtrans_hat))

    prob = prob_1 * prob_2 * prob_3
    return prob


def angle_abs_pi(angle1, angle2):
    """Calculates the difference of two angles and return the value between [-pi,pi]"""
    diff = angle1 - angle2
    if diff > pi or diff < -pi:
        diff_sign = np.sign(diff)
        diff = diff - diff_sign * 2 * pi

    return diff


def sample_motion_model_odometry(ut, xt_1, odom_std, N_particles=1):
    """ Calculates a vector of sampled robot positions xt based on odometry measure ut and
    last known position xt_1.
    ut: array[6] of odometry positions in instants (t-1) and (t)
    xt_1: array[3] of known position of single particle or array([3,N_particles]) for more particles
    odom_std: array[4] of standard deviations in odometry measures
    N_particles: number of particles
    """

    a1 = odom_std[0]
    a2 = odom_std[1]
    a3 = odom_std[2]
    a4 = odom_std[3]

    x_bar = ut[0]
    y_bar = ut[1]
    theta_bar = ut[2]
    x_bar_l = ut[3]
    y_bar_l = ut[4]
    theta_bar_l = ut[5]

    if len(xt_1.shape) == 1:
        x = np.copy([xt_1[0]])
        y = np.copy([xt_1[1]])
        theta = np.copy([xt_1[2]])
    else:
        x = np.copy(xt_1[0, :])
        y = np.copy(xt_1[1, :])
        theta = np.copy(xt_1[2, :])

    dtrans = np.sqrt((x_bar_l - x_bar) ** 2 + (y_bar_l - y_bar) ** 2)
    drot1 = np.arctan2(y_bar_l - y_bar, x_bar_l - x_bar) - theta_bar
    drot2 = theta_bar_l - theta_bar - drot1

    drot1_hat = (a1 * abs(drot1) + a2 * abs(dtrans)) * np.random.randn(N_particles) + drot1
    dtrans_hat = (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2))) * np.random.randn(N_particles) + dtrans
    drot2_hat = (a1 * abs(drot2) + a2 * abs(dtrans)) * np.random.randn(N_particles) + drot2

    x_l = x + dtrans_hat * np.cos(theta + drot1_hat)
    y_l = y + dtrans_hat * np.sin(theta + drot1_hat)
    theta_l = theta + drot1_hat + drot2_hat

    xt_samples = np.stack((x_l, y_l, theta_l))

    return xt_samples


def landmark_model_correspondence(landmark_measure, landmark_corresp, xt, landmark_map, landmark_std, N_particles=1):
    """ Calculates the density probability function of the robot measurements (landmark_measure,landmark_corresp) of landmarks
    given the particles positions vector xt and map landmark_map of landmarks.
       landmark_measure: array([2, N_landmarks]) landmark position measured (r:distance, phi:angle)
       landmark_corresp: array([N_landmarks]) of correspondences between real and measured landmarks
       xt: array([3,N_particles]) positions of particles before incorporates sensor measures
       landmark_map: array([2,N_landmarks]) of landmarks positions
       landmark_std: array([2]) of standard deviation extraction landmarks measures
       N_particles = number of particles
         """

    r_std = landmark_std[0]
    phi_std = landmark_std[1]

    #Transform xt array into column
    if len(xt.shape) == 1:
        xt.resize((3,1))

    p_land_temp = np.ones((landmark_map.shape[1], N_particles), dtype=float)
    r_hat = np.zeros((landmark_map.shape[1], N_particles), dtype=float)
    phi_hat = np.zeros((landmark_map.shape[1], N_particles), dtype=float)

    for k in range(0, landmark_map.shape[1], 1):
        j = int(landmark_corresp[k])
        mxj = landmark_map[0, j - 1]
        myj = landmark_map[1, j - 1]
        r_hat[j - 1, :] = np.sqrt((mxj - xt[0, :]) ** 2 + (myj - xt[1, :]) ** 2)
        phi_hat[j - 1, :] = np.arctan2(myj - xt[1, :], mxj - xt[0, :]) - xt[2, :]
        p_land_temp[j - 1, :] = norm.pdf(landmark_measure[0, j - 1] - r_hat[j - 1, :], loc=0, scale=r_std) * norm.pdf(
            landmark_measure[1, j - 1] - phi_hat[j - 1, :], loc=0, scale=phi_std)

    prob_landmark = np.prod(p_land_temp, axis=0)
    return prob_landmark


def sample_landmark_model_correspondence(fi, ci, m, landmark_var, N_particles):
    """Function sample_landmark_model_correspondence(fi, ci, m, landmark_var): Samples positions based on
    landmark measurements and correspondents variances"""

    r_i = fi[0]
    phi_i = fi[1]
    r_var = landmark_var[0]
    phi_var = landmark_var[1]
    gamma_hat = np.random.rand(N_particles)*2*np.pi
    r_hat = r_i + np.random.randn(N_particles)*r_var
    phi_hat = phi_i + np.random.randn(N_particles)*phi_var
    x = m[0] + r_hat*np.cos(gamma_hat)
    y = m[1] + r_hat*np.sin(gamma_hat)
    theta = gamma_hat - np.pi - phi_hat
    if N_particles == 1:
        xt_samples = np.array((x,y,theta))
    else:
        xt_samples = np.stack((x,y,theta))

    return xt_samples


def low_variance_sampler(Xt, N_particles):
    """ Low Variance Resampling method for sampling particles Xt based on weighing (weight = Xt[3] 4th element)
     of each particle
     Xt: array([4,N_particles]) of particles positions and weights in instant t
     N_particles : number of particles"""

    Xbart = np.zeros((4, N_particles))
    r = 0 + (1 / N_particles - 0) * np.random.rand()

    w1t = Xt[3, 0]
    c = w1t
    i = 1
    for m in range(1, N_particles, 1):
        u = r + (m - 1) * (1 / N_particles)
        while (u > c):
            i = i + 1
            c = c + Xt[3, i - 1]
        Xbart[:, m - 1] = np.copy(Xt[:, i - 1])

    return Xbart


def particle_filter_algorithm(Xt_1, N_particles, ut, landmark_map, zt, odom_std, landmark_std):
    """ Executes the Localization particle filter algorithm and returns the position of each particle
    after incorporating odometry and sensors measurements.
    Xt_1: array([4,N_particles]) of particles positions and weights in instant t-1
    N_particles: Number of particles
    ut: array[6] of odometry positions odom(t-1) and odom(t)
    landmark_map: array([2,N_landmarks]) of landmarks positions
    zt: array(3,N_landmarks) of landmarks measures and correspondences
    """

    fi = np.copy(zt[0:2, :])
    ci = np.copy(zt[2, :])

    n = 0.0
    Xmt = sample_motion_model_odometry(ut, Xt_1, odom_std, N_particles)
    Wmt = landmark_model_correspondence(fi, ci, Xmt, landmark_map, landmark_std, N_particles)
    n = np.sum(Wmt)
    Wmt = Wmt / n
    Xbart = np.vstack((Xmt, Wmt))
    Xtresult = low_variance_sampler(Xbart, N_particles)
    return Xtresult


def create_vehicle_command(u_command, time_hold, t_vec, dt):
    """Creates a array of linear and angular velocities for a vehicle (robot) to be used during simulation of
    Particle Filter localization.
    u_comand = array(2,n) - eg. [[10, 10, 10],[pi, pi, pi]]
    time_hold = array(1,n) in seconds - eg. [[5, 5, 5]]
    t_vec = (seconds) array of each step time simulation
    dt = (seconds) step time of simulation
    """

    t_len = len(t_vec)
    u_len = u_command.shape[1]
    total_time = 0.0
    id_old = -1
    repeat = np.array([], dtype=int)
    last_i = 0
    for i in range(u_len):
        total_time = total_time + time_hold[i]
        if id_old + 1 < t_len:
            div_len = int(time_hold[i] / dt)
            repeat = np.append(repeat, div_len)
            id_old = id_old + div_len
            last_i = i

    repeat[last_i] = repeat[last_i] + t_len - id_old - 1
    u_vec = np.repeat(u_command[:, 0:last_i + 1], repeat, axis=1)
    return u_vec


def particle_filter_simulation(x0_in, odom_std_in, landmark_std_in, N_particles, T, dt, animate, plot_axes, plot_obj):
    """Encapsulates a example of particle localization simulation. Creates a map and simulates a mobile robot
    x0_in: array[3] initial robot position in the map
    odom_std_in: array[4] of standard deviations in odometry measures
    landmark_std_in: array[2] of standard deviation landmarks extraction measures
    N_particles: number of particles
    T: Time of simulation
    dt: interval time of simulation
    animate: if equal to "animate" runs an plot animation of estimated positions evolution over time
    plot_axes: axes for plotting data
    plot_obj: plot object of matplot lib to be used in animation
    """

    x0 = np.array(x0_in)
    xreal_old = np.copy(x0)
    xodot1 = np.copy(x0)
    xodot_1 = np.copy(x0)

    odom_std = np.array(odom_std_in)
    landmark_std = np.array(landmark_std_in)

    #Landmarks positons map
    m = np.stack((np.concatenate((np.arange(5), np.arange(5))), np.concatenate((np.ones(5) * 2, np.ones(5) * 3)),
                  np.linspace(1, 10, 10)))  # map of environment (landmarks)

    mapxend = 8
    mapyend = 6

    #Generate random particles over the environment to start estimation
    xinit = -0.5 + (mapxend - (-0.5)) * np.random.rand(N_particles)
    yinit = -0.5 + (mapyend - (-0.5)) * np.random.rand(N_particles)
    thetainit = np.zeros((N_particles), dtype=float)

    X0 = np.stack((xinit, yinit, thetainit, np.zeros(N_particles)))

    t = np.arange(0, T, dt)

    #Creates angular and linear velocities for the mobile robot movement over the entire simulation time
    v0 = 0.15
    w0 = 20 * pi / 180
    u = np.array(((v0, v0, v0, v0), (w0, -w0, w0, -w0)))
    time_hold = np.array([5, 5, 5, 5])

    u_vec = create_vehicle_command(u, time_hold, t, dt)

    #set plot parameters and plots created landmarks map
    plot_axes.set_xlim([-2, 5])
    plot_axes.set_ylim([-2, 5])
    col = ['r', 'g', 'b', 'c', 'm', 'k']
    col_len = len(col)
    plot_axes.scatter(m[0, :], m[1, :], c='w', s=120, marker='^', edgecolors='g')

    #Arrays to store current/anterior particles positons and weights
    Xt = np.copy(X0)
    Xtold = np.copy(Xt)
    k = 1

    t_len = t.size

    #Arrays to store data during simulation
    xodo_salva = np.zeros((3, t_len), dtype=float)
    xreal_salva = np.zeros((3, t_len), dtype=float)
    Xt_pos_salva = np.zeros((3, 2, N_particles), dtype=float)
    xhat_salva = np.zeros((3, t_len), dtype=float)

    start_time = time.time()

    for i in range(0, t_len, 1):

        xodot1[0] = xodot_1[0] + u_vec[0, i] * dt * np.cos(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[1] = xodot_1[1] + u_vec[0, i] * dt * np.sin(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[2] = xodot_1[2] + u_vec[1, i] * dt

        ut = np.concatenate((xodot_1, xodot1))

        xreal = sample_motion_model_odometry(ut, xreal_old, odom_std)
        xreal.resize(3)

        #Computes landmarks meausres
        zt = np.stack((np.sqrt((m[0, :] - xreal[0]) ** 2 + (m[1, :] - xreal[1]) ** 2) + landmark_std[0] * np.random.randn(),
                       np.arctan2(m[1, :] - xreal[1], m[0, :] - xreal[0]) - xreal[2] + landmark_std[1] * np.random.randn(),
                       m[2, :]))
        
        Xt = particle_filter_algorithm(Xtold, N_particles, ut, m, zt, odom_std_in, landmark_std_in)

        #Estimates robot position based on particles
        x_hat = np.average(Xt[0:3, :], weights=Xt[3, :], axis=1)
        Xtold = np.copy(Xt)

        #Plot estimated particles positions over determined intervals
        if t[i] % 2 == 0:
            plot_axes.scatter(Xt[0, :], Xt[1, :], 2, col[int(k % (col_len - 1))], zorder=3)
            if (Xt_pos_salva.shape[0] == k):
                Xt_pos_salva = np.concatenate((Xt_pos_salva, np.reshape(Xt[0:2, :], (1, 2, N_particles))), axis=0)
            else:
                Xt_pos_salva[k, :, :] = np.copy(Xt[0:2, :])
            k = k + 1

        xodot_1 = np.copy(xodot1)
        xreal_old = np.copy(xreal)
        xodo_salva[:, i] = np.copy(xodot1)
        xreal_salva[:, i] = np.copy(xreal)
        xhat_salva[:, i] = np.copy(x_hat)

        print("Percentage executed: %.2f %%" % (t[i] * 100 / T))

    dtime = time.time() - start_time
    print("Execution time: %.2f seconds" % (dtime))

    plot_axes.plot(xodo_salva[0, :], xodo_salva[1, :], c='r', label='x odometry')

    if animate == "animate":
        line, = plot_axes.plot([], [], lw=2, c='g')
        #anim = animation.FuncAnimation(plot_canvas_in.fig, animation_test, fargs=(xreal_salva, axesplot_in, line), frames=t_len, interval=20)
    else:
        plot_axes.scatter(xreal_salva[0, :], xreal_salva[1, :], s=10, c='w', edgecolors='g')
        plot_axes.plot(xreal_salva[0, :], xreal_salva[1, :], c='g', label='x real')
        plot_axes.plot(xhat_salva[0, :], xhat_salva[1, :], c='b', label='x estimated')
    plot_axes.legend()
    plot_obj.show()
