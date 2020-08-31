# Script for executing Particle Filter Simulation
import particle_filter.particlefilter as pfilter
import matplotlib.pyplot as plt

if __name__ == "__main__":
   np = pfilter.np
   fig1, ax1 = plt.subplots()
   x0 = np.array((0.0,0.0,0.0))
   odom_variance = [0.6, 0.4, 0.4, 0.4]
   landmark_variance = [0.2, 0.2, 0.1]
   M_particles= 500
   T = 20
   dt = 0.1
   pfilter.particle_filter_simulation(x0, odom_variance, landmark_variance, M_particles, T, dt, "", ax1, plt)