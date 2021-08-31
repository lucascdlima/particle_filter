# Script for executing Particle Filter Simulation
import particle_filter.particlefilter as pfilter
import matplotlib.pyplot as plt

if __name__ == "__main__":
   np = pfilter.np
   fig1, ax1 = plt.subplots()
   x0 = np.array((0.0,0.0,0.0))
   #odom_std = [1.8, 0.5, 0.05, 0.02]
   #landmark_std = [0.2, 0.2]

   odom_std = [0.1, 0.01, 0.02, 0.01]
   #landmark_std = [0.05, 0.05]
   landmark_std = [0.1, 0.1]

   M_particles= 500
   T = 20
   dt = 0.1
   pfilter.particle_filter_simulation(x0, odom_std, landmark_std, M_particles, T, dt, "animate", ax1, fig1, plt)
