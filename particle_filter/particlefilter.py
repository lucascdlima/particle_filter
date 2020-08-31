# Particle filter module
import numpy as np
from numpy import pi
from scipy.stats import norm
import time


def prob_normal_distribution(x, bvar):
    """Function prob_normal_distribution(x, bvar) returns a density probability function of a normal distribution
    (mean = 0, bvar = standard deviation) evaluated at the point x. """

    norm_prob = np.exp(-0.5 * ((x / bvar) ** 2)) / (np.sqrt(2 * pi) * bvar)
    return norm_prob


def motion_model_odometry(xt, ut, xt_1, alpha):
    """Function motion_model_odometry(xt, ut, xt_1, alpha) returns a density probability function of
      a particle being in position xt based on the odometry measure ut and last position known xt_1.
      alpha = vector of standard deviation in odometry measures"""

    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]
    a4 = alpha[3]

    xbar = ut[0]
    ybar = ut[1]
    thetabar = ut[2]
    xbarl = ut[3]
    ybarl = ut[4]
    thetabarl = ut[5]

    x = xt_1[0]
    y = xt_1[1]
    theta = xt_1[2]

    xl = xt[0]
    yl = xt[1]
    thetal = xt[2]

    dtrans = np.sqrt((xbarl - xbar) ** 2 + (ybarl - ybar) ** 2)
    drot1 = np.arctan2(ybarl - ybar, xbarl - xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    dtranshat = np.sqrt((xl - x) ** 2 + (yl - y) ** 2)
    drot1hat = np.arctan2(yl - y, xl - x) - theta
    drot2hat = thetal - theta - drot1hat

    diff_rot1 = drot1 - drot1hat
    diff_rot2 = drot2 - drot2hat

    if drot1 - drot1hat > pi or drot1 - drot1hat < -pi:
        diff_sign = np.sign(drot1 - drot1hat)
        diff_rot1 = drot1 - drot1hat - diff_sign * 2 * pi
        print("entrou diff rot1 > pi ")

    if drot2 - drot2hat > pi or drot2 - drot2hat < -pi:
        diff_sign = np.sign(drot2 - drot2hat)
        diff_rot2 = drot2 - drot2hat - diff_sign * 2 * pi
        print("entrou diff rot2 > pi ")

    p1 = prob_normal_distribution(diff_rot1, a1 * abs(drot1hat) + a2 * abs(dtranshat))
    p2 = prob_normal_distribution(dtrans - dtranshat, a3 * abs(dtranshat) + a4 * (abs(drot1hat) + abs(drot2hat)))
    p3 = prob_normal_distribution(diff_rot2, a1 * abs(drot2hat) + a2 * abs(dtranshat))

    p = p1 * p2 * p3
    return p


def angle_abs_pi(drot1,drot1hat):
    if drot1 - drot1hat > pi or drot1 - drot1hat < -pi:
        diff_sign = np.sign(drot1 - drot1hat)
        diff_rot1 = drot1 - drot1hat - diff_sign*2*pi
        #print(f'diff total = {(drot1 - drot1hat)*180/pi} degree')
        #print(f'diff abs = {diff_rot1*180/pi} degree ')


def sample_motion_model_odometry(ut, xt_1, alpha):
    """ Function sample_motion_model_odometry(ut, xt_1, alpha): returns a sampled robot position xt
    based on the odometry measure ut and last position known xt_1.
    alpha: array[4] standard deviation elements in odometry measures.
    ut: array[6] of odometry positions odom(t-1) and odom(t)
     """

    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]
    a4 = alpha[3]

    xbar = ut[0]
    ybar = ut[1]
    thetabar = ut[2]
    xbarl = ut[3]
    ybarl = ut[4]
    thetabarl = ut[5]

    x = xt_1[0]
    y = xt_1[1]
    theta = xt_1[2]

    dtrans = np.sqrt((xbarl - xbar) ** 2 + (ybarl - ybar) ** 2)
    drot1 = np.arctan2(ybarl - ybar, xbarl - xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    drot1hat = drot1 + (a1 * abs(drot1) + a2 * abs(dtrans)) * np.random.randn()
    dtranshat = dtrans + (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2))) * np.random.randn()
    drot2hat = drot2 + (a1 * abs(drot2) + a2 * abs(dtrans)) * np.random.randn()

    xl = x + dtranshat * np.cos(theta + drot1hat)
    yl = y + dtranshat * np.sin(theta + drot1hat)
    thetal = theta + drot1hat + drot2hat

    xt = np.array([xl, yl, thetal])
    return xt


def sample_motion_model_odometry_vec(ut, xt_1, alpha, M):
    """ Function sample_motion_model_odometry_vec(ut, xt_1, alpha,M) returns a vector of
    sampled robot (particles) positions xt based on the odometry measure ut and last position known xt_1.
    alpha = vector of standard deviation in measures.
    M = number of particles.
    """

    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]
    a4 = alpha[3]

    xbar = ut[0]
    ybar = ut[1]
    thetabar = ut[2]
    xbarl = ut[3]
    ybarl = ut[4]
    thetabarl = ut[5]

    x = np.copy(xt_1[0, :])
    y = np.copy(xt_1[1, :])
    theta = np.copy(xt_1[2, :])

    dtrans = np.sqrt((xbarl - xbar) ** 2 + (ybarl - ybar) ** 2)
    drot1 = np.arctan2(ybarl - ybar, xbarl - xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    drot1hat = (a1 * abs(drot1) + a2 * abs(dtrans)) * np.random.randn(M) + drot1
    dtranshat = (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2))) * np.random.randn(M) + dtrans
    drot2hat = (a1 * abs(drot2) + a2 * abs(dtrans)) * np.random.randn(M) + drot2

    xl = x + dtranshat * np.cos(theta + drot1hat)
    yl = y + dtranshat * np.sin(theta + drot1hat)
    thetal = theta + drot1hat + drot2hat

    xt_odo = np.stack((xl, yl, thetal))
    return xt_odo


def landmark_model_correspondence(fi, ci, x, m, landmark_var):
    """ Function landmark_model_correspondence(fi,ci,x,m,var) returns the density probability function
     of the robot measurements (fi,ci) of landmarks given the robot position x and map m of landmarks.
     landmark_var = vector of standard deviation of measures.
       """

    var1 = landmark_var[0]
    var2 = landmark_var[1]

    p_land = 1
    for k in range(0, m.shape[1], 1):
        j = int(ci[k])
        mxj = m[0, j - 1]
        myj = m[1, j - 1]
        rbar = np.sqrt((mxj - x[0]) ** 2 + (myj - x[1]) ** 2)
        phibar = np.arctan2(myj - x[1], mxj - x[0]) - x[2]

        p_land = p_land * prob_normal_distribution(rbar - fi[0, j - 1], var1) * prob_normal_distribution(
            phibar - fi[1, j - 1], var2)

    return p_land


def landmark_model_correspondence_vec(fi, ci, x, m, landmark_var, M):
    """ Function landmark_model_correspondence(fi,ci,x,m,var) returns a vector of density probability functions
       of the robot measurements (fi,ci) of landmarks given the particles robot positions vector x and map m of landmarks.
       var = vector of standard deviation of measures
       M = number of particles
         """
    var1 = landmark_var[0]
    var2 = landmark_var[1]

    p_land_temp = np.ones((m.shape[1], M), dtype=float)
    r_hat = np.zeros((m.shape[1], M), dtype=float)
    phi_hat = np.zeros((m.shape[1], M), dtype=float)

    for k in range(0, m.shape[1], 1):
        j = int(ci[k])
        mxj = m[0, j - 1]
        myj = m[1, j - 1]
        r_hat[j - 1, :] = np.sqrt((mxj - x[0, :]) ** 2 + (myj - x[1, :]) ** 2)
        phi_hat[j - 1, :] = np.arctan2(myj - x[1, :], mxj - x[0, :]) - x[2, :]
        p_land_temp[j - 1, :] = norm.pdf(fi[0, j - 1] - r_hat[j - 1, :], loc=0, scale=var1) * norm.pdf(fi[1, j - 1] - phi_hat[j - 1, :], loc=0, scale=var2)

    p_land = np.prod(p_land_temp, axis=0)
    return p_land


def low_variance_sampler(Xt, M):
    """Function low_variance_sampler(Xt, M): Resampling of particles Xt based on weighting (weight = Xt[3] 4th element)
     of each particle
     M = number of particles"""

    Xbart = np.zeros((4, M))
    r = 0 + (1 / M - 0) * np.random.rand()

    w1t = Xt[3, 0]
    c = w1t
    i = 1
    for m in range(1, M, 1):
        u = r + (m - 1) * (1 / M)
        while (u > c):
            i = i + 1
            c = c + Xt[3, i - 1]
        Xbart[:, m - 1] = np.copy(Xt[:, i - 1])
    #Added normalization step (confirma if is correct)
    #n = np.sum( Xbart[3,:])
    #Xbart[3,:] = Xbart[3,:]/n
    return Xbart


def particle_filter_algorithm(Xt_1, M, ut, m, zt, odom_variance, landmark_varince):
    """Function particle_filter_algorithm(Xt_1,M,ut,m,zt,alpha,varsens): Execute the Localization particle filter algorithm
    which returns the position of each particle after odometry and sensors measurements are incorporated"""

    fi = np.copy(zt[0:2, :])
    ci = np.copy(zt[2, :])

    n = 0.0
    Xmt = sample_motion_model_odometry_vec(ut, Xt_1, odom_variance, M)
    Wmt = landmark_model_correspondence_vec(fi, ci, Xmt, m, landmark_varince, M)
    n = np.sum(Wmt)
    Wmt = Wmt / n
    Xbart = np.vstack((Xmt, Wmt))
    Xtresult = low_variance_sampler(Xbart, M)
    return Xtresult


def create_vehicle_command(u_command, time_hold, t_vec, dt):
    """Function create_vehicle_command (u_command, time_hold, t_vec, dt) creates a array of linear and angular velocities
    for a vehicle (robot) to be used during simulation of Particle Filter localization
    u_comand = array(2,n) - eg. [[10, 10, 10],[pi, pi, pi]]
    time_hold = array(1,n) in seconds - eg. [[5, 5, 5]]
    t_vec = (seconds) array of each step time simulation
    dt = (seconds) step time of simulation
    """

    t_len = len(t_vec)
    u_len = u_command.shape[1]
    total_time = 0.0
    id_old = -1
    repeat = np.array([],dtype=int)
    last_i = 0
    for i in range(u_len):
        total_time = total_time + time_hold[i]
        if id_old+1<t_len:
            div_len = int(time_hold[i]/dt)
            repeat = np.append(repeat,div_len)
            id_old = id_old + div_len
            last_i = i

    repeat[last_i] = repeat[last_i] + t_len - id_old - 1
    u_vec = np.repeat(u_command[:,0:last_i+1],repeat,axis=1)
    return u_vec


def particle_filter_simulation(x0_in, odovar_in, landmarkvar_in, M, T, dt, animate, axes, plot_obj):

    x0 = np.array(x0_in)
    xreal = np.copy(x0)
    xreal_old = np.copy(x0)
    xodot1 = np.copy(x0)
    xodot_1 = np.copy(x0)

    # alpha = [0.07 0.05 0.05 0.05]; %moving aprox 1 m, errors = aprox 0.1 m
    #alpha = np.array([0.8, 0.6, 0.6, 0.6])  # variance of odometric readings
    #varsen = np.array([0.15, 0.15, 0.1])  # variance of sensor landmark readings
    alpha = np.array(odovar_in)  # variance of odometric readings
    varsen = np.array(landmarkvar_in)  # variance of sensor landmark readings

    m = np.stack((np.concatenate((np.arange(5), np.arange(5))), np.concatenate((np.ones(5) * 2, np.ones(5) * 3)),
                  np.linspace(1, 10, 10)))  # map of environment (landmarks)

    mapxend = 8
    mapyend = 6

    xinit = -0.5 + (mapxend - (-0.5)) * np.random.rand(M)
    yinit = -0.5 + (mapyend - (-0.5)) * np.random.rand(M)
    thetainit = np.zeros((M),dtype=float)

    X0 = np.stack((xinit, yinit, thetainit, np.zeros(M)))

    t = np.arange(0, T, dt)

    v0 = 0.15
    w0 = 20 * pi / 180
    u = np.array(((v0, v0, v0, v0), (w0, -w0, w0, -w0)))
    time_hold = np.array([5, 5, 5, 5])

    u_vec = create_vehicle_command(u, time_hold, t, dt)

    axes.set_xlim([-2, 5])
    axes.set_ylim([-2, 5])

    col = ['r', 'g', 'b', 'c', 'm', 'k']
    col_len = len(col)
    axes.scatter(m[0, :], m[1, :], c='w', s=120, marker='^', edgecolors='g')

    Xt = np.copy(X0)
    Xtold = np.copy(Xt)
    k = 1

    t_len = t.size

    xodo_salva = np.zeros((3, t_len),dtype=float)
    xreal_salva = np.zeros((3, t_len),dtype=float)
    Xt_pos_salva = np.zeros((3,2,M),dtype=float)
    xhat_salva = np.zeros((3, t_len),dtype=float)

    start_time = time.time()

    for i in range(0, t_len, 1):

        xodot1[0] = xodot_1[0] + u_vec[0, i] * dt * np.cos(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[1] = xodot_1[1] + u_vec[0, i] * dt * np.sin(xodot_1[2] + u_vec[1, i] * dt)
        xodot1[2] = xodot_1[2] + u_vec[1, i] * dt

        ut = np.concatenate((xodot_1, xodot1))
        xreal = sample_motion_model_odometry(ut, xreal_old, alpha)

        zt = np.stack((np.sqrt((m[0, :] - xreal[0]) ** 2 + (m[1, :] - xreal[1]) ** 2) + varsen[0] * np.random.randn(),
                       np.arctan2(m[1, :] - xreal[1], m[0, :] - xreal[0]) - xreal[2] + varsen[1] * np.random.randn(),
                       m[2, :]))

        Xt = particle_filter_algorithm(Xtold, M, ut, m, zt, odovar_in, landmarkvar_in)
        x_hat = np.average(Xt[0:3,:], weights = Xt[3,:], axis = 1)
        Xtold = np.copy(Xt)

        if t[i] % 2 == 0:
            axes.scatter(Xt[0, :], Xt[1, :], 2, col[int(k%(col_len-1))], zorder=3)
            if(Xt_pos_salva.shape[0]==k):
                Xt_pos_salva = np.concatenate((Xt_pos_salva,np.reshape(Xt[0:2,:],(1,2,M))),axis=0)
            else:
                pass
                Xt_pos_salva[k,:,:] = np.copy(Xt[0:2,:])
            k = k + 1

        xodot_1 = np.copy(xodot1)
        xreal_old = np.copy(xreal)
        xodo_salva[:, i] = np.copy(xodot1)
        xreal_salva[:, i] = np.copy(xreal)
        xhat_salva[:, i] = np.copy(x_hat)

        print("Percentual exec: %f " % (t[i] * 100 / T))

    dtime = time.time() - start_time
    print("execution time: %s seconds" % (dtime))

    axes.plot(xodo_salva[0, :], xodo_salva[1, :], c='r', label='x odometry')


    if animate == "animate":
        line, = axes.plot([], [], lw=2, c='g')
        #anim = animation.FuncAnimation(plot_canvas_in.fig, animation_test, fargs=(xreal_salva, axesplot_in, line), frames=t_len, interval=20)
    else:
        axes.scatter(xreal_salva[0, :], xreal_salva[1, :], s=10, c='w', edgecolors='g')
        axes.plot(xreal_salva[0, :], xreal_salva[1, :], c='g', label='x real')
        axes.plot(xhat_salva[0, :], xhat_salva[1, :], c='b', label='x estimated')
    axes.legend()
    plot_obj.show()
