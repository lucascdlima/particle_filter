
#Particle filter module
import numpy as np
from numpy import pi
from scipy.stats import norm

def prob_normal_distribution(xnorm, bvar):
    """prob_normal_distribution(xnorm, bvar) returns a density probability function of a normal distribution
    (zero - mean, bvar - standard deviation) evaluated at the point xnorm. """

    norm_prob = np.exp(-0.5*((xnorm/bvar)**2))/(np.sqrt(2*pi)*bvar)
    return norm_prob


def motion_model_odometry(xt, ut, xt_1, alpha):
    """Function motion_model_odometry(xt, ut, xt_1, alpha) returns a density probability function of
      a particle being in position xt based on the odometry measure ut and last position known xt_1.
      Parameter alpha = vector of standard deviation in measures"""
    
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

    dtrans = np.sqrt((xbarl - xbar)**2 + (ybarl-ybar)**2)
    drot1 = np.arctan2(ybarl-ybar,xbarl-xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    dtranshat = np.sqrt((xl - x)**2 + (yl-y)**2)
    drot1hat = np.arctan2(yl-y,xl-x) - theta
    drot2hat = thetal - theta - drot1hat

    p1 = prob_normal_distribution(drot1-drot1hat,a1*abs(drot1hat) + a2*abs(dtranshat))
    p2 = prob_normal_distribution(dtrans-dtranshat,a3*abs(dtranshat) + a4*(abs(drot1hat)+abs(drot2hat)))
    p3 = prob_normal_distribution(drot2-drot2hat,a1*abs(drot2hat) + a2*abs(dtranshat))

    p = p1*p2*p3
    return p


def sample_motion_model_odometry(ut, xt_1, alpha):
    """ Function sample_motion_model_odometry(ut, xt_1, alpha) returns a sampled robot position xt
    based on the odometry measure ut and last position known xt_1.
    alpha = vector of standard deviation in measures.
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

    dtrans = np.sqrt((xbarl - xbar)**2 + (ybarl-ybar)**2)
    drot1 = np.arctan2(ybarl-ybar,xbarl-xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    drot1bar = drot1 + (a1 * abs(drot1) + a2 * abs(dtrans))*np.random.randn()
    dtransbar = dtrans + (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2)))*np.random.randn()
    drot2bar = drot2 + (a1 * abs(drot2) + a2 * abs(dtrans))*np.random.randn()

    xl = x + dtransbar * np.cos(theta + drot1bar)
    yl = y + dtransbar * np.sin(theta + drot1bar)
    thetal = theta + drot1bar + drot2bar

    xt = np.array([xl, yl, thetal])
    return xt

def sample_motion_model_odometry_vec(ut, xt_1, alpha,M):
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

    x = np.copy(xt_1[0,:])
    y = np.copy(xt_1[1,:])
    theta = np.copy(xt_1[2,:])

    dtrans = np.sqrt((xbarl - xbar)**2 + (ybarl-ybar)**2)
    drot1 = np.arctan2(ybarl-ybar,xbarl-xbar) - thetabar
    drot2 = thetabarl - thetabar - drot1

    drot1bar = (a1 * abs(drot1) + a2 * abs(dtrans))*np.random.randn(M) + drot1
    dtransbar = (a3 * abs(dtrans) + a4 * (abs(drot1) + abs(drot2)))*np.random.randn(M) + dtrans
    drot2bar = (a1 * abs(drot2) + a2 * abs(dtrans))*np.random.randn(M) + drot2

    xl = x + dtransbar * np.cos(theta + drot1bar)
    yl = y + dtransbar * np.sin(theta + drot1bar)
    thetal = theta + drot1bar + drot2bar

    xt_odo = np.stack((xl, yl, thetal))
    return xt_odo


def landmark_model_correspondence(fi,ci,x,m,var):
    """ Function landmark_model_correspondence(fi,ci,x,m,var) returns the density probability function
     of the robot measurements (fi,ci) of landmarks given the robot position x and map m of landmarks.
     var = vector of standard deviation of measures.
       """

    var1 = var[0]
    var2 = var[1]

    p_land = 1
    for k in range(0, m.shape[1], 1):
        j = int(ci[k])
        mxj = m[0,j-1]
        myj = m[1,j-1]
        rbar = np.sqrt((mxj - x[0])**2 + (myj - x[1])**2)
        phibar = np.arctan2(myj - x[1],mxj - x[0]) - x[2]

        p_land = p_land*prob_normal_distribution(rbar - fi[0,j-1], var1)*prob_normal_distribution(phibar - fi[1,j-1], var2)

    return p_land


def landmark_model_correspondence_vec(fi,ci,x,m,var,M):
    """ Function landmark_model_correspondence(fi,ci,x,m,var) returns a vector of density probability functions
       of the robot measurements (fi,ci) of landmarks given the particles robot positions vector x and map m of landmarks.
       var = vector of standard deviation of measures
       M = number of particles
         """
    var1 = var[0]
    var2 = var[1]

    p_land_temp = np.ones((m.shape[1],M),dtype=float)
    rbar = np.zeros((m.shape[1],M),dtype=float)
    phibar = np.zeros((m.shape[1],M),dtype=float)

    for k in range(0, m.shape[1], 1):
        j = int(ci[k])
        mxj = m[0,j-1]
        myj = m[1,j-1]
        rbar[j-1,:] = np.sqrt((mxj - x[0,:])**2 + (myj - x[1,:])**2)
        phibar[j-1,:] = np.arctan2(myj - x[1,:],mxj - x[0,:]) - x[2,:]
        p_land_temp[j-1,:] = norm.pdf(rbar[j-1,:] - fi[0,j-1], loc=0, scale=var1)*norm.pdf(phibar[j-1,:] - fi[1,j-1], loc=0, scale=var2)

    p_land = np.prod(p_land_temp,axis=0)
    return p_land


def low_variance_sampler(Xt, M):
    """Function low_variance_sampler(Xt, M): Resampling of particles Xt based on weighting (weight = Xt(3))
     of each particle
     M = number of particles"""

    Xbart = np.zeros((4, M))
    r = 0 + (1 / M - 0) * np.random.rand()

    w1t = Xt[3, 0]
    c = w1t
    i = 1
    for m in range(1,M,1):
        u = r + (m - 1) * (1 / M)
        while (u > c):
            i = i + 1
            c = c + Xt[3, i-1]
        Xbart[:, m-1] = np.copy(Xt[:, i-1])

    return Xbart


def particle_filter_algorithm(Xt_1,M,ut,m,zt,alpha,varsens):
    """Function particle_filter_algorithm(Xt_1,M,ut,m,zt,alpha,varsens): Particle filter algorithm which returns
    the position of each particle after odometry and sensors measurements are incorporated"""
    fi = np.copy(zt[0:2,:])
    ci = np.copy(zt[2,:])

    n = 0.0
    Xmt = sample_motion_model_odometry_vec(ut,Xt_1,alpha,M)
    Wmt = landmark_model_correspondence_vec(fi,ci,Xmt,m,varsens,M)
    n = np.sum(Wmt)
    Wmt = Wmt/n
    Xbart = np.vstack((Xmt,Wmt))
    Xtresult = low_variance_sampler(Xbart, M)
    return Xtresult
