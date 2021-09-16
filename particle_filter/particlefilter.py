# Particle filter module
import numpy as np
from numpy import pi
from scipy.stats import norm
import time
from math import modf
from matplotlib import animation
import matplotlib.pyplot as plt

scatter_particles = None
def motion_model_odometry(xt, ut, xt_1, odom_std):
    """Calculates a density probability function of a particle in position xt given the known
    odometry measure ut and last position xt_1. In other words, computes the probability p(xt/ut,xt_1).
      Args:
          xt: array((3)) current unknown position.
          odom_std = array((4)) of standard deviations in odometry measures.
          ut: array((6)) of odometry positions in instants (t-1) and (t).
          xt_1: array((3)) of known position in instant t-1.
      Returns:
            probability density function value, otherwise returns None if odom_std values are zero.
      """

    a1 = odom_std[0]
    a2 = odom_std[1]
    a3 = odom_std[2]
    a4 = odom_std[3]

    if((a1==0 and a2==0) or (a3==0 and a4==0)):
        return None

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
    """Calculates the difference of two angles (angle1 - angle2) and return the value between [-pi,pi].
    Angles are expressed in radians"""
    
    diff = angle1 - angle2
    fractional, whole = modf(diff/(2.0*pi))
    diff = fractional*2.0*pi
    if diff > pi or diff < -pi:
        diff_sign = np.sign(diff)
        diff = diff - diff_sign * 2.0 * pi

    return diff

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def angle_diff(angle1, angle2):
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    d1 = angle1 - angle2
    d2 = 2*np.pi - abs(d1)

    scalar=True
    if(np.isscalar(angle2) == False):
        if(angle2.size > 1):
            scalar = False
            filter_greater = np.greater(d1,np.zeros(d1.size))*1
            filter_no_greater = 1 - filter_greater
            d2 = np.multiply(d2,-filter_greater) + np.multiply(d2,filter_no_greater)

            d2_greater_d1 = np.greater(abs(d2),abs(d1))*1
            not_d2_greater_d1 = 1 - d2_greater_d1
            result = d2_greater_d1*d1 + not_d2_greater_d1*d2
            return result
    if(scalar):
        if (d1 > 0):
            d2 *= -1.0
        if (abs(d1) < abs(d2)):
            return (d1)
        else:
            return (d2)



def sample_motion_model_odometry(ut, xt_1, odom_std, N_particles=1):
    """ Calculates a vector of sampled robot positions xt based on odometry measure ut and
    last known position xt_1.
    Args:
        ut: array((6)) of odometry positions in instants (t-1) and (t)
        xt_1: array((3,N_particles)) of known position of single particle or array([3,N_particles]) for more particles
        odom_std: array((4)) of standard deviations in odometry measures
        N_particles: number of particles
    Returns:
        array((3,N_particles)) of sampled positions with (x,y,phi) coordinates.
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

    # if(dtrans < 0.01):
    #     drot1 = 0.0
    # else:
    #     drot1 = np.arctan2(y_bar_l - y_bar, x_bar_l - x_bar) - theta_bar
    #
    # drot2 = theta_bar_l - theta_bar - drot1

    # TODO: TRUNCATE THE ANGLES DIFFERENCES, ANGLES SUM TO?
    # #Avoid computing distance rotation if traslation is small (just rotation movements)
    if(dtrans < 0.01):
        drot1 = 0.0
    else:
       drot1 = angle_diff(np.arctan2(y_bar_l - y_bar, x_bar_l - x_bar), theta_bar)

    drot2 = angle_diff(angle_diff(theta_bar_l, theta_bar), drot1)

    sigma_rot1 = (a1 * abs(drot1) + a2 * abs(dtrans))
    sigma_rot2 = (a1 * abs(drot2) + a2 * abs(dtrans))
    sigma_trans = (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2)))

    noise_drot1 = sigma_rot1 * np.random.randn(N_particles)
    noise_drot2 = sigma_rot2 * np.random.randn(N_particles)
    noise_dtrans = sigma_trans * np.random.randn(N_particles)

    drot1_hat = angle_diff(drot1, noise_drot1)
    dtrans_hat = -noise_dtrans + dtrans
    drot2_hat = angle_diff(drot2, noise_drot2)

    # drot1_hat = np.array([angle_diff(drot1, value) for value in noise_drot1])
    # dtrans_hat = -(a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2))) * np.random.randn(N_particles) + dtrans
    # drot2_hat = np.array([angle_diff(drot2, value) for value in noise_drot2])

    # #TODO: CHECK IF NEED TO USE SQRT OF NOISE PARAMETERS
    # drot1_hat = -(a1 * abs(drot1) + a2 * abs(dtrans)) * np.random.randn(N_particles) + drot1
    # dtrans_hat = -(a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2))) * np.random.randn(N_particles) + dtrans
    # drot2_hat = -(a1 * abs(drot2) + a2 * abs(dtrans)) * np.random.randn(N_particles) + drot2

    x_l = x + dtrans_hat * np.cos(theta + drot1_hat)
    y_l = y + dtrans_hat * np.sin(theta + drot1_hat)
    theta_l = theta + drot1_hat + drot2_hat

    xt_samples = np.stack((x_l, y_l, theta_l))
    std_noise_values = np.stack((sigma_trans, sigma_rot1, sigma_rot2))


    return xt_samples , std_noise_values


def landmark_model_correspondence(landmark_measure, landmark_corresp, xt, landmark_map, landmark_std, N_particles=1):
    """ Calculates the density probability function of the robot measurements (landmark_measure,landmark_corresp) of landmarks
    given the particles positions vector xt and map landmark_map of landmarks.
       Args:
            landmark_measure: array((2, N_landmarks)) landmark position measured (r:distance, phi:angle)
            landmark_corresp: array((N_landmarks)) of correspondences between real and measured landmarks
            xt: array((3,N_particles)) positions of particles before incorporates sensor measures
            landmark_map: array((2,N_landmarks)) of landmarks positions
            landmark_std: array((2)) of standard deviation extraction landmarks measures
            N_particles = number of particles
        Returns:
            array((1,N_particles)) of probabilities values

         """

    r_std = landmark_std[0]
    phi_std = landmark_std[1]

    #Transform xt array into column
    if len(xt.shape) == 1:
        xt.resize((3,1))

    if(landmark_measure is None):
        return None

    N_land_mes = landmark_measure.shape[1]
    p_land_temp = np.ones((N_land_mes, N_particles), dtype=float)
    r_hat = np.zeros((N_land_mes, N_particles), dtype=float)
    phi_hat = np.zeros((N_land_mes, N_particles), dtype=float)

    for k in range(N_land_mes):
        j = int(landmark_corresp[k])
        mxj = landmark_map[0, j]
        myj = landmark_map[1, j]
        r_hat[k, :] = np.sqrt((mxj - xt[0, :]) ** 2 + (myj - xt[1, :]) ** 2)
        phi_hat[k, :] = np.arctan2(myj - xt[1, :], mxj - xt[0, :]) - xt[2, :]  #prob robotics book is missing the subtraction of (theta)
        # p_land_temp[k, :] = norm.pdf(landmark_measure[0, k] - r_hat[k, :], loc=0, scale=r_std) * norm.pdf(
        #     landmark_measure[1, k] - phi_hat[k, :], loc=0, scale=phi_std)
        p_land_temp[k, :] = norm.pdf(r_hat[k, :] - landmark_measure[0, k], loc=0, scale=r_std) * norm.pdf(
            phi_hat[k, :] - landmark_measure[1, k], loc=0, scale=phi_std)

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

#TODO: STUDY THE RESAMPLING METHODS
def low_variance_sampler(Xt, N_particles):
    """ Low Variance Resampling method for sampling particles Xt based on weighing (weight = Xt[3] 4th element)
     of each particle.
     Args:
         Xt: array((4,N_particles)) of particles positions and weights in instant t
         N_particles : number of particles
     Returns:
         array((4,N_particles)) with particles positions (array[0:3, index]) and respective weights [array[3, index]]
         """

    Xbart = np.zeros((4, N_particles),dtype=float)
    # r = 0.0 + (1.0 / N_particles - 0) * np.random.rand()
    r = np.random.uniform(0, 1.0/N_particles)

    w1t = Xt[3, 0] #first particle_0 weight
    c = w1t
    i = 1
    for m in range(1, N_particles+1, 1):
        u = r + (m - 1) * (1.0/N_particles)
        while (u > c):
            i = i + 1
            c = c + Xt[3, i - 1]
        Xbart[:, m - 1] = np.copy(Xt[:, i - 1])
        # Xbart[3, m - 1] = 1/N_particles

    return Xbart


def particle_filter_algorithm(Xt_1, N_particles, ut, landmark_map, zt, odom_std, landmark_std):
    """ Run the Localization particle filter algorithm and returns the position of each particle
    after incorporating odometry and sensors measurements.
    Args:
        Xt_1: array((4,N_particles)) of particles positions and weights in instant t-1
        N_particles: Number of particles
        ut: array((6)) of odometry positions odom(t-1) and odom(t)
        landmark_map: array([2,N_landmarks]) of landmarks positions
        zt: array((3,N_landmarks)) of landmarks measures and correspondences
    Returns:
        array((4,N_particles)) of particles with positions and weights updated
    """

    fi = np.copy(zt[0:2, :])
    ci = np.copy(zt[2, :])

    n = 0.0
    Xmt, noise_odom = sample_motion_model_odometry(ut, Xt_1, odom_std, N_particles)
    """Wmt = landmark_model_correspondence(fi, ci, Xmt, landmark_map, landmark_std, N_particles)
    n = np.sum(Wmt)
    Wmt = Wmt / n
    Xbart = np.vstack((Xmt, Wmt))
    Xtresult = low_variance_sampler(Xbart, N_particles)

    print(n)"""
    if(zt is None):
        Wmt = np.ones((1,N_particles))/N_particles
        Xbart = np.vstack((Xmt, Wmt))
        Xtresult = np.copy(Xbart)
    else:
        fi = np.copy(zt[0:2, :])
        ci = np.copy(zt[2, :])

        Wmt = landmark_model_correspondence(fi, ci, Xmt, landmark_map, landmark_std, N_particles)
        # n = np.sum(Wmt)
        # Wmt = Wmt / n
        # Xbart = np.vstack((Xmt, Wmt))

        if(check_motion(ut)):
            n = np.sum(Wmt)
            Wmt = Wmt / n
            Xbart = np.vstack((Xmt, Wmt))
            Xtresult = low_variance_sampler(Xbart, N_particles)
        else:
            Xbart = np.vstack((Xmt, Wmt))
            Xtresult = np.copy(Xbart)
            Wmt_new = np.multiply(Xt_1[3,:],Wmt)
            Xtresult[3, :] = np.copy(Wmt_new/sum(Wmt_new))

    return Xtresult, noise_odom


def create_vehicle_command(u_command, time_hold, t_vec, dt):
    """Creates a array of linear and angular velocities for a vehicle (robot) to be used during simulation of
    Particle Filter localization.
    Args:
        u_command = array((2,n)) - eg. [[10, 10, 10],[pi, pi, pi]]
        time_hold = array((1,n)) in seconds - eg. [[5, 5, 5]]
        t_vec = (seconds) array of each step time simulation
        dt = (seconds) step time of simulation
    Returns:
        array((2,columns)) of linear and angular velocities applied in each step time of simulation.
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


def particle_filter_simulation(x0_in, odom_std_in, landmark_std_in, N_particles, T, dt, animate, plot_axes, plot_fig, plot_obj, noisy_map, simulation_type):
    """Encapsulates a example of particle localization simulation. Creates a map and simulates a mobile robot
    Args:
        x0_in: array((3)) initial robot position in the map
        odom_std_in: array((4)) of standard deviations in odometry measures
        landmark_std_in: array((2)) of standard deviation landmarks extraction measures
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
    # number_landamarks = 10
    # m = np.stack((np.concatenate((np.arange(5), np.arange(5))), np.concatenate((np.ones(5,dtype=float) * 2, np.ones(5,dtype=float) * 3)),
    #               np.linspace(0, 9, 10)))  # map of environment (landmarks)
    # m[1,:] = m[1,:] + np.array([0.5, 0.8, 0.1, 2, 0.2, 1, 1.5, 0.5, 1.0, 0.2])

    ##Simple test with smal numbers of land marks ---------------------------------------------------------------------------------------------
    # m = np.array([[2.0, 3.0, 4.0],[1.0,3.0,2.0],[0,1,2]])
    #

    t = np.arange(0, T, dt)

    if(simulation_type == 'simple'):
        m = np.array([[2.0, 4.0, 6.0], [1.0, 6.0, 4.0], [0, 1, 2]])
        m_noisy = np.copy(m)
        if (noisy_map):
            m_noisy[1, 2] = m_noisy[1, 2] + 1.0

        # Creates angular and linear velocities for the mobile robot movement over the entire simulation time
        v0 = 0.15
        w0 = 10.0 * pi / 180
        u = np.array(
            (np.ones(12) * v0,
             (-w0, w0, -w0, w0, -w0, w0, 1.5*w0, -w0, w0, -w0,w0, -w0)))

        time_hold = np.ones(12)*int(T/12)
        landmarks_window = m.shape[1]
        mapxend = 10.0
        mapyend = 10.0
        particles_x_max = 3.0
        particles_y_max = 5.0
        u_vec = create_vehicle_command(u, time_hold, t, dt)

    elif(simulation_type == 'long'):
        m = create_landmark_map()

        m_noisy = np.copy(m)

        # Creates angular and linear velocities for the mobile robot movement over the entire simulation time
        v0 = 0.45
        w0 = 5.0 * pi / 180

        u = np.array(
            (np.ones(20)*v0, (-w0, w0, -w0, w0, -w0, w0, w0, w0, -w0, w0, -w0, w0, -w0, w0, -w0, w0, -w0, w0, -w0, w0)))

        time_hold = np.ones(20)*int(T/20)

        u_vec = create_vehicle_command(u, time_hold, t, dt)

        landmarks_window = 5
        mapxend = 25.0
        mapyend = 25.0
        particles_x_max = 3.0
        particles_y_max = 5.0
        paths = []
        path1 = dict(pos = [3.0, 3.5], u= [0,1.0], distance = 17.0 , phi = 0.0, end=[20.5,3.5])
        paths.append(path1)
        path2 = dict(pos=[20.5, 3.5], u=[-1.0, 0.0], distance=16.0, phi = np.pi/2, end=[20.5,20.5])
        paths.append(path2)
        path3 = dict(pos=[20.5, 20.5], u=[0, -1.0], distance=17.0, phi = np.pi, end=[3.0,20.5])
        paths.append(path3)
        n_paths = 3
        k_path = 0

        path = path1
    plot_axes.scatter(m[0, :], m[1, :], c='w', s=120, marker='^', edgecolors='g', label='Landmarks map', zorder=2)
    if(noisy_map):
        plot_axes.scatter(m_noisy[0, :], m_noisy[1, :], c='w', s=120, marker='^', edgecolors='r',label='Noisy Landmark map', zorder=1)

    # plot_obj.show()



    #Generate random particles over the environment to start estimation
    xinit = -1.0 + (particles_x_max - (-1.0)) * np.random.rand(N_particles)
    yinit = -1.0 + (particles_y_max - (-1.0)) * np.random.rand(N_particles)
    thetainit = np.zeros((N_particles), dtype=float)
    #thetainit = -np.pi + (np.pi - (-np.pi)) * np.random.rand(N_particles)

    X0 = np.stack((xinit, yinit, thetainit, np.zeros(N_particles, dtype=float)))

    # N_p = int(np.sqrt(N_particles))
    # xinit = np.linspace(0, mapxend, int(np.sqrt(N_particles)))
    # yinit = np.linspace(0, mapyend, int(np.sqrt(N_particles)))
    # xinit, yinit = np.meshgrid(xinit, yinit,sparse=False, indexing='xy')
    # XYinit = np.array([xinit.flatten(), yinit.flatten()]) + np.array((-0.2 + (0.2 - (-0.2)) * np.random.rand(N_particles),-0.2 + (0.2 - (-0.2)) * np.random.rand(N_particles)))

    # scatter_dots = plot_axes.scatter(XYinit[0, :], XYinit[1, :], 2, c='b', zorder=3)
    # X0 = np.vstack((XYinit, thetainit.reshape((1,N_particles)), np.zeros(N_particles, dtype=float)))

    #set plot parameters and plots created landmarks map
    plot_axes.set_xlim([-2, mapxend])
    plot_axes.set_ylim([-2, mapyend])
    col = ['r', 'g', 'b', 'c', 'm', 'k']
    col_len = len(col)



    #Arrays to store current/anterior particles positons and weights
    Xt = np.copy(X0)
    Xtold = np.copy(Xt)
    k = 0

    t_len = t.size

    #Arrays to store data during simulation
    xodo_save = np.zeros((3, t_len), dtype=float)
    xreal_save = np.zeros((3, t_len), dtype=float)
    Xt_pos_save = np.zeros((4, N_particles,t_len), dtype=float)
    xhat_save = np.zeros((3, t_len), dtype=float)
    xerro_save = np.zeros((3, t_len), dtype=float)
    noisy_trees_save = np.zeros((3, t_len), dtype=float)
    robot_control_save = np.zeros((2, t_len), dtype=float)
    std_noise_odom_save = np.zeros((3, t_len), dtype=float)
    X_particles_save = []

    start_time = time.time()


    noisy_indice = -1.0
    for i in range(0, t_len, 1):

        if(simulation_type == 'long'):

            if(np.linalg.norm(xreal_old[0:2] - path['end'])<=2.0 and k_path<n_paths-1):
                k_path += 1
                path = paths[k_path]

            # if(xreal_old[1]>=20.0):
            #     path = path3
            wc = robot_control(xreal_old,path,v0,1,1)

            xodot1[0] = xodot_1[0] + v0 * dt * np.cos(xodot_1[2] + wc * dt)
            xodot1[1] = xodot_1[1] + v0 * dt * np.sin(xodot_1[2] + wc * dt)
            xodot1[2] = xodot_1[2] + wc * dt
            v_control = [v0, wc]
        else:
            xodot1[0] = xodot_1[0] + u_vec[0, i] * dt * np.cos(xodot_1[2] + u_vec[1, i] * dt)
            xodot1[1] = xodot_1[1] + u_vec[0, i] * dt * np.sin(xodot_1[2] + u_vec[1, i] * dt)
            xodot1[2] = xodot_1[2] + u_vec[1, i] * dt

            v_control = [u_vec[0, i], u_vec[1, i]]

        ut = np.concatenate((xodot_1, xodot1))

        xreal, temp  = sample_motion_model_odometry(ut, xreal_old, odom_std)
        xreal.resize(3)

        #Computes simulated landmarks measures with noise

        map_match = check_nearest_landmarks(xreal,m, landmarks_window)

        if(simulation_type=='long' and noisy_map):
            noisy_angle = -pi/2 + pi*np.random.rand()
            # map_match[0, 0] = map_match[0, 0] + np.cos(noisy_angle)
            # map_match[1, 0] = map_match[1, 0] + np.sin(noisy_angle)
            # noisy_trees_save[:,i] = np.copy(map_match[0:2,0])
            if(noisy_indice not in map_match[2,:]):
                id = np.random.randint(landmarks_window-1)
                map_match[0, id] = map_match[0, id] + np.cos(noisy_angle)
                map_match[1, id] = map_match[1, id] + np.sin(noisy_angle)
                noisy_trees_save[:,i] = np.copy(map_match[:,id])
                noisy_indice = map_match[2,id]
            else:
                id = np.where(map_match[2,:] == noisy_indice)
                map_match[:, id[0]] = np.copy(np.reshape(noisy_trees_save[:, i-1], (3,1)))
                noisy_trees_save[:, i] = np.copy(noisy_trees_save[:, i-1])

        zt = np.stack(
            (np.sqrt((map_match[0, :] - xreal[0]) ** 2 + (map_match[1, :] - xreal[1]) ** 2) + landmark_std[0] * np.random.randn(),
             np.arctan2(map_match[1, :] - xreal[1], map_match[0, :] - xreal[0]) - xreal[2] + landmark_std[1] * np.random.randn(),
             map_match[2, :]))

        # zt = np.stack((np.sqrt((m[0, :] - xreal[0]) ** 2 + (m[1, :] - xreal[1]) ** 2) + landmark_std[0] * np.random.randn(),
        #                np.arctan2(m[1, :] - xreal[1], m[0, :] - xreal[0]) - xreal[2] + landmark_std[1] * np.random.randn(),
        #                m[2, :]))
        
        Xt, std_noise_odom= particle_filter_algorithm(Xtold, N_particles, ut, m_noisy, zt, odom_std_in, landmark_std_in)

        #Estimates robot position based on particles
        x_hat = np.average(Xt[0:3, :], weights=Xt[3, :], axis=1)
        Xtold = np.copy(Xt)

        xodot_1 = np.copy(xodot1) #checked
        xreal_old = np.copy(xreal)
        xodo_save[:, i] = np.copy(xodot1)
        xreal_save[:, i] = np.copy(xreal)
        xhat_save[:, i] = np.copy(x_hat)
        xerro_save[:,i] = np.copy([xreal[0] - x_hat[0],xreal[1] - x_hat[1],angle_diff(xreal[2], x_hat[2])])
        Xt_pos_save[:,:,i]= np.copy(Xt)
        robot_control_save[:,i] = np.copy(v_control)
        std_noise_odom_save[:,i] = np.copy(std_noise_odom)

        print("Percentage executed: %.2f %%" % (t[i] * 100 / T))

    dtime = time.time() - start_time
    print("Execution time: %.2f seconds" % (dtime))

    #plot_axes.plot(xodo_save[0, :], xodo_save[1, :], c='r', label='x odometry')

    if animate == "animate":
       # fig_animation, ax = plt.figure()

        line_position_real, = plot_axes.plot([], [], lw=2, c='g')
        line_position_odom, = plot_axes.plot([], [], lw=2, c='r')
        line_position_pf, = plot_axes.plot([], [], lw=2, c='b')
        # line_landmarks, = plot_axes.scatter([], [], s=10, c='w', edgecolors='g')
        plot_data = []
        plot_indices = {'xreal':0,'xodo':1,'xhat':2,'Xt':3}
        plot_data.append(xreal_save)
        plot_data.append(xodo_save)
        plot_data.append(xhat_save)
        plot_data.append(Xt_pos_save)
        # plot_data.append(landmarks_map)

        # line_data = []
        # line_data.append(line_position_real)
        # line_data.append(xodo_save)
        # line_data.append(xhat_save)
        # line_data.append(landmarks_map)

       # anim = animation.FuncAnimation(plot_canvas_in.fig, animation_test2, fargs=(xreal_save, axesplot_in, line), frames=t_len, interval=20)
        anim = animation.FuncAnimation(plot_fig, animation_test2, fargs=(plot_data, plot_axes, plot_indices),
                                       frames=t_len, interval=20)
    elif(animate=='step'):
        plot_obj.ion()
        plot_obj.show()
        scatter_dots = plot_axes.scatter(X0[0, :], X0[1, :], 2, c='b', zorder=3)
        for i in range(0, t_len, 1):
            # plot_fig.canvas.draw()
            # plot_fig.canvas.flush_events()
            scatter_dots.remove()
            scatter_dots = plot_axes.scatter(Xt_pos_save[0, :,i], Xt_pos_save[1, :,i], s=2, c='tab:orange', zorder=1)
            plot_axes.scatter(xreal_save[0, i], xreal_save[1, i], s=2, c='w', edgecolors='g',zorder=4)
            plot_axes.scatter(xodo_save[0, i], xodo_save[1, i], s=2, c='r', label='x real',zorder=2)
            plot_axes.scatter(xhat_save[0, i], xhat_save[1, i], s=2, c='b', label='x estimated',zorder=3)



    else:
        plot_axes.scatter(xreal_save[0, :], xreal_save[1, :], s=2, c='g', label='x real',zorder=4)
        plot_axes.scatter(xodo_save[0, :], xodo_save[1, :], s=2, c='r', label='x odometry', zorder=2)
        plot_axes.scatter(xhat_save[0, :], xhat_save[1, :], s=2, c='b', label='x estimated', zorder=3)
        if (noisy_map):
            plot_axes.scatter(noisy_trees_save[0, :], noisy_trees_save[1, :], c='w', s=120, marker='^', edgecolors='r', zorder=1)

        fontsize = 'xx-large'
        plot_axes.set_ylabel('y (m)', fontsize=fontsize)
        plot_axes.set_xlabel('x (m)', fontsize=fontsize)
        plot_axes.tick_params(axis='both',labelsize=fontsize)
        fig2, ax2 = plot_obj.subplots(3,1)
        ax2[0].plot(t,xerro_save[0,:],label='Error x position')
        ax2[0].set_ylabel('Error x (m)', fontsize=fontsize)
        ax2[0].set_xlabel('time(s)', fontsize=fontsize)
        ax2[0].grid(True)
        ax2[0].tick_params(axis='both', labelsize=fontsize)
        ax2[1].plot(t, xerro_save[1, :], label='Error y position')
        ax2[1].set_ylabel('Error y (m)', fontsize=fontsize)
        ax2[1].set_xlabel('time(s)', fontsize='x-large')
        ax2[1].grid(True)
        ax2[1].tick_params(axis='both', labelsize=fontsize)
        ax2[2].plot(t, xerro_save[2, :], label='Error theta - orientation')
        ax2[2].set_ylabel('Error theta (rad)', fontsize=fontsize)
        ax2[2].set_xlabel('time(s)', fontsize=fontsize)
        ax2[2].grid(True)

        fig3, ax3 = plot_obj.subplots(2, 1)
        ax3[0].tick_params(axis='both', labelsize=fontsize)
        ax3[0].plot(t, robot_control_save[0, :], label='Robot linear velocity')
        ax3[0].set_ylabel('vc (m/s)', fontsize=fontsize)
        ax3[0].set_xlabel('time(s)', fontsize=fontsize)
        ax3[0].grid(True)
        ax3[0].tick_params(axis='both', labelsize=fontsize)

        ax3[1].tick_params(axis='both', labelsize=fontsize)
        ax3[1].plot(t, robot_control_save[1, :], label='Robot angular velocity')
        ax3[1].set_ylabel('wc (rad/s)', fontsize=fontsize)
        ax3[1].set_xlabel('time(s)', fontsize=fontsize)
        ax3[1].grid(True)
        ax3[1].tick_params(axis='both', labelsize=fontsize)

        fig4, ax4 = plot_obj.subplots(3, 1)
        ax4[0].tick_params(axis='both', labelsize=fontsize)
        ax4[0].set_ylabel('std_trans: m', fontsize=fontsize)
        ax4[0].set_xlabel('time(s)', fontsize=fontsize)
        ax4[0].grid(True)
        ax4[0].tick_params(axis='both', labelsize=fontsize)

        ax4[1].tick_params(axis='both', labelsize=fontsize)
        ax4[1].set_ylabel('std_rot1: rad', fontsize=fontsize)
        ax4[1].set_xlabel('time(s)', fontsize=fontsize)
        ax4[1].grid(True)
        ax4[1].tick_params(axis='both', labelsize=fontsize)

        ax4[2].tick_params(axis='both', labelsize=fontsize)
        ax4[2].set_ylabel('std_rot2: rad', fontsize=fontsize)
        ax4[2].set_xlabel('time(s)', fontsize=fontsize)
        ax4[2].grid(True)
        ax4[2].tick_params(axis='both', labelsize=fontsize)

        ax4[0].scatter(t, std_noise_odom_save[0,:], label='Odometry: translation std deviation')
        ax4[1].scatter(t, std_noise_odom_save[1,:], label='Odometry: rotation 1 std deviation')
        ax4[2].scatter(t, std_noise_odom_save[2, :], label='Odometry: rotation 2 std deviation')


        odom_std = np.array(odom_std_in)
        landmark_std = np.array(landmark_std_in)
        print('Simulation parameters:')
        print('Time (s): {}'.format(T))
        print('Time interval dt (s): {}'.format(dt))
        print('Odometry measures - std deviation coefficients (sigma1, sigma2, sigma3, sigma4): {} {} {} {}'.format(odom_std[0],odom_std[1],odom_std[2],odom_std[3]))
        print('Odometry measures - std deviation average: translation - {:2.4} (m), rotation 1 - {:2.4} (rad), rotation 2 - {:2.4} (rad)'.format(
            np.mean(std_noise_odom_save[0,:]), np.mean(std_noise_odom_save[1,:]), np.mean(std_noise_odom_save[2,:])))

        print('Landmarks measures - std deviation (sigma_r, sigma_phi): {} (m) , {} (rad)'.format(landmark_std[0],landmark_std[1]))
        print('Robot linear and angular velocities: v = {} m/s , w = {} rad/s'.format(v0,w0))

        print('test mean rot2 = ',np.mean(std_noise_odom_save[2,:]))

        # arrow_dx = 0.01 * np.cos(xreal_save[2, :])
        # arrow_dy = 0.01 * np.sin(xreal_save[2, :])
        # plot_axes.quiver(xreal_save[0, :], xreal_save[1, :], arrow_dx, arrow_dy, scale=0.5)

    # plot_axes.legend(fontsize=fontsize)
    plot_obj.show()

def check_motion(ut):
    x_bar = ut[0]
    y_bar = ut[1]
    theta_bar = ut[2]

    x_bar_l = ut[3]
    y_bar_l = ut[4]
    theta_bar_l = ut[5]

    dtrans = np.sqrt((x_bar_l - x_bar) ** 2 + (y_bar_l - y_bar) ** 2)
    dtheta = angle_diff(theta_bar_l,theta_bar)
    if(dtrans>0.01 or abs(dtheta) > 5.0*np.pi/180.0):
        return True

    return False

def animation_test2(i, data_plot_in, ax_in, plot_indices):
    global scatter_particles
    if(scatter_particles is not None):
        scatter_particles.remove()
    ax_in.plot(data_plot_in[plot_indices['xreal']][0,0:i], data_plot_in[plot_indices['xreal']][1,0:i], lw=2, c='g')
    ax_in.plot(data_plot_in[plot_indices['xodo']][0,0:i], data_plot_in[plot_indices['xodo']][1,0:i], lw=2, c='r')
    ax_in.plot(data_plot_in[plot_indices['xhat']][0,0:i], data_plot_in[plot_indices['xhat']][1,0:i], lw=2, c='b')
    scatter_particles = ax_in.scatter(data_plot_in[plot_indices['Xt']][0, :,i], data_plot_in[plot_indices['Xt']][1, :,i], s=2, c='tab:orange')
    # ax_in.scatter(data_plot_in[4][0, :], data_plot_in[4][1, :], s=10, c='w', edgecolors='g')

def create_landmark_map():
    line1 = np.array(([2, 20],[3, 3]))
    line2 = np.array(([20, 20], [3, 20]))
    line3 = np.array(([20, 2], [20, 20]))
    map_x = np.array([0])
    map_y = np.array([0])

    for i in np.arange(line1[0,0],line1[0,1],1.5):

        map_x = np.append(map_x, i + np.random.rand())
        map_y = np.append(map_y, line1[1,0] + 1.0 + np.random.rand()*0.8)

        map_x = np.append(map_x, i + np.random.rand())
        map_y = np.append(map_y, line1[1, 0] - 1.0 + np.random.rand() * 0.8)

    for i in np.arange(line2[1, 0], line2[1, 1], 1.5):
        map_x = np.append(map_x, line2[0, 0] + 1.0 + np.random.rand() * 0.8)
        map_y = np.append(map_y, i + np.random.rand())

        map_x = np.append(map_x, line2[0, 0] - 1.0 + np.random.rand() * 0.8)
        map_y = np.append(map_y, i + np.random.rand())

    for i in np.arange(line3[0, 0], line3[0, 1], -1.5):
        map_x = np.append(map_x,i + np.random.rand())
        map_y = np.append(map_y,line3[1, 0] + 1.0 + np.random.rand() * 0.8)

        map_x = np.append(map_x, i + np.random.rand())
        map_y = np.append(map_y, line3[1, 0] - 1.0 + np.random.rand() * 0.8)

    N_landmarks = map_x.size
    map_landmarks = np.array((map_x, map_y, range(N_landmarks)))
    return map_landmarks


def check_nearest_landmarks(xreal, landmarks_map, landmarks_window):
    N_landmarks = landmarks_map.shape[1]

    distances = np.zeros(N_landmarks, dtype=float)
    for i in range(N_landmarks):
        distances[i] = np.sqrt((landmarks_map[0, i] - xreal[0]) ** 2 + (landmarks_map[1, i] - xreal[1]) ** 2)

    indices = sorted(range(len(distances)), key=lambda sub: distances[sub])[:landmarks_window]


    landmarks_map_corresp = landmarks_map[:,indices]

    return landmarks_map_corresp

def robot_control(x_real, path, vc, k2, k1):

    l = path['u'][0]*(x_real[0] - path['pos'][0]) + path['u'][1]*(x_real[1] - path['pos'][1])
    err_phi = angle_diff(x_real[2],path['phi'])

    if(abs(err_phi)>=0.001):
        wc = -k2*vc*l*np.sin(err_phi)/err_phi - k1*err_phi
    else:
        wc = -k2 * vc * l * 1 - k1 * err_phi
    return wc