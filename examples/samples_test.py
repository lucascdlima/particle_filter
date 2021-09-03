import particle_filter.particlefilter as pfilter
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

  fig1, ax1 = plt.subplots()
  x0 = np.array((0.0,0.0,0.0))

  # odom_std = [0.1, 0.15, 0.2, 0.01]

  odom_std = [0.1, 0.1, 0.2, 0.1] # odom_std = [0.1, 0.1, 0.2, 0.1] one possibility of set up odometry sigmas (sigma2 seems to spread points)
  # odom_std = [0.1, 0.15, 0.2, 0.01]

  # odom_std = [0.1, 0.3, 0.2, 0.1]
  landmark_variance = [0.2, 0.2, 0.2]
  M_particles= 1000
  #T = 20
  T = 4
  dt = 0.1
  #pfilter.particle_filter_simulation(x0, odom_variance, landmark_variance, M_particles, T, dt, "", ax1, plt)

  xodot1 = np.copy(x0)
  xodot_1 = np.copy(x0)
  # v0 = 0.15
  v0 = 0.4
  #w0 = 20 * np.pi / 180
  w0 = 0
  t1 = np.arange(0, 40, dt)
  t2 = np.arange(0, 40, dt)
  t3 = np.arange(0, 80, dt)
  total_len = len(t1) + len(t2) + len(t3)
  xodo_salva = np.zeros((3, total_len), dtype=float)
  Xt_1 = np.stack((np.zeros(M_particles, dtype=float), np.zeros(M_particles, dtype=float), np.zeros(M_particles, dtype=float)))
  xodot_1 = np.copy(x0)

  for i in range(0, len(t1), 1):
     xodot1[0] = xodot_1[0] + v0 * dt * np.cos(xodot_1[2] + w0 * dt)
     xodot1[1] = xodot_1[1] + v0 * dt * np.sin(xodot_1[2] + w0 * dt)
     xodot1[2] = xodot_1[2] + w0 * dt
     xodo_salva[:, i] = np.copy(xodot1)
     ut = np.concatenate((xodot_1, xodot1))
     Xt = pfilter.sample_motion_model_odometry(ut, Xt_1, odom_std, M_particles)
     xodot_1 = np.copy(xodot1)
     if((i*dt) % 20 == 0):
         ax1.scatter(Xt[0, :], Xt[1, :], s=2, c='r', label='x real particles')
         # arrow_dx = 0.01 * np.cos(Xt[2, :])
         # arrow_dy = 0.01 * np.sin(Xt[2, :])
         # ax1.quiver(Xt[0, :], Xt[1, :], arrow_dx, arrow_dy, width=0.01)
     Xt_1 = np.copy(Xt)

  xodot_1[2] = np.pi/2
  Xt_1[2,:] = Xt_1[2,:] + np.pi/2
  for i in range(len(t1),len(t1)+len(t2) , 1):
      xodot1[0] = xodot_1[0] + v0 * dt * np.cos(xodot_1[2] + w0 * dt)
      xodot1[1] = xodot_1[1] + v0 * dt * np.sin(xodot_1[2] + w0 * dt)
      xodot1[2] = xodot_1[2] + w0 * dt
      xodo_salva[:, i] = np.copy(xodot1)
      ut = np.concatenate((xodot_1, xodot1))
      Xt = pfilter.sample_motion_model_odometry(ut, Xt_1, odom_std, M_particles)
      xodot_1 = np.copy(xodot1)
      if ((i * dt) % 20 == 0):
          ax1.scatter(Xt[0, :], Xt[1, :], s=2, c='r', label='x real particles')
          # arrow_dx = 0.01 * np.cos(Xt[2, :])
          # arrow_dy = 0.01 * np.sin(Xt[2, :])
          # ax1.quiver(Xt[0, :], Xt[1, :], arrow_dx, arrow_dy, width=0.01)
      Xt_1 = np.copy(Xt)

  xodot_1[2] = np.pi
  Xt_1[2, :] = Xt_1[2, :] + np.pi / 2
  for i in range(len(t2), len(t2) + len(t3), 1):
      xodot1[0] = xodot_1[0] + v0 * dt * np.cos(xodot_1[2] + w0 * dt)
      xodot1[1] = xodot_1[1] + v0 * dt * np.sin(xodot_1[2] + w0 * dt)
      xodot1[2] = xodot_1[2] + w0 * dt
      xodo_salva[:, i] = np.copy(xodot1)
      ut = np.concatenate((xodot_1, xodot1))
      Xt = pfilter.sample_motion_model_odometry(ut, Xt_1, odom_std, M_particles)
      xodot_1 = np.copy(xodot1)
      if ((i * dt) % 20 == 0):
          ax1.scatter(Xt[0, :], Xt[1, :], s=2, c='r', label='x real particles')
          # arrow_dx = 0.01 * np.cos(Xt[2, :])
          # arrow_dy = 0.01 * np.sin(Xt[2, :])
          # ax1.quiver(Xt[0, :], Xt[1, :], arrow_dx, arrow_dy, width=0.01)
      Xt_1 = np.copy(Xt)

  ax1.plot(xodo_salva[0,:], xodo_salva[1,:], c='b', label='xodo')



  fig2, ax2 = plt.subplots()

  landmark_variance = [0.3, 0.3]

  m = np.array([3,4])
  fi = np.array([5,np.arctan2(4,3)])
  xt_samples = pfilter.sample_landmark_model_correspondence(fi,0,m,landmark_variance,M_particles)
  ax2.scatter(xt_samples[0, :], xt_samples[1, :], c='r', label='x samples', zorder=2)
  ax2.scatter(3,4, c='b', label='Landmark')
  arrow_dx = 0.01*np.cos(xt_samples[2,:])
  arrow_dy = 0.01*np.sin(xt_samples[2,:])
  ax2.quiver(xt_samples[0, :], xt_samples[1, :],arrow_dx,arrow_dy,width = 0.01, zorder=1)
  ax2.set_aspect('equal', adjustable='datalim')
  plt.show()
