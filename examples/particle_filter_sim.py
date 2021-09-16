# Script for executing Particle Filter Simulation
import particle_filter.particlefilter as pfilter
import matplotlib.pyplot as plt

if __name__ == "__main__":
   np = pfilter.np
   fig1, ax1 = plt.subplots()
   x0 = np.array((1.0,4.0,0.0))

   odom_std = [0.1, 0.1, 0.2, 0.1]
   landmark_std = [0.3, 0.3] #last used deviation
   landmark_std = [0.8, 0.5]

   # odom_std = [1.8, 0.5, 0.05, 0.02]
   # landmark_std = [0.2, 0.2]

   # odom_std = [0.025, 0.025, 0.4, 0.4]

   # odom_std = [0.1, 0.01, 0.02, 0.01]
   # #landmark_std = [0.05, 0.05]
   # landmark_std = [0.1, 0.1]

   M_particles= 2000
   T = 90
   dt = 0.1
   noisy_map = True
   pfilter.particle_filter_simulation(x0, odom_std, landmark_std, M_particles, T, dt, "", ax1, fig1, plt, noisy_map, 'long')
