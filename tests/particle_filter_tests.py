import unittest
import particle_filter.particlefilter as pfilter
import numpy as np


class ParticleFilterTest(unittest.TestCase):

    def setUp(self):
        self.x0 = np.array((0.0, 0.0, 0.0))
        self.xodot1 = np.copy(self.x0)
        self.xodot_1 = np.copy(self.x0)
        self.v0 = 0.15
        self.w0 = 10.0 * np.pi / 180
        self.dt = 0.5
        self.xodot1[0] = self.xodot_1[0] + self.v0 * self.dt * np.cos(self.xodot_1[2] + self.w0 * self.dt)
        self.xodot1[1] = self.xodot_1[1] + self.v0 * self.dt * np.sin(self.xodot_1[2] +self. w0 * self.dt)
        self.xodot1[2] = self.xodot_1[2] + self.w0 * self.dt
        self.ut = np.concatenate((self.xodot_1, self.xodot1))

    def test_motion_model_odometry(self):

        xt = self.xodot1 + np.array((0.01,0.01,3*np.pi/180))
        odom_std = [0.1, 0.01, 0.02, 0.01]

        msg_error = "Probability value cannot be less or equal zero"
        self.assertGreaterEqual(pfilter.motion_model_odometry(xt,self.ut,self.x0,odom_std), 0, msg=msg_error)

        msg_error = "If odometry standard deviations are equal zero (eg. [0,0,2,1] or [1,3,0,0])"\
                    "function must return None"
        odom_std = [0, 0, 0.02, 0.01]
        self.assertIsNone(pfilter.motion_model_odometry(xt, self.ut, self.x0, odom_std),msg=msg_error)

        odom_std = [1, 2, 0, 0]
        self.assertIsNone(pfilter.motion_model_odometry(xt, self.ut, self.x0, odom_std),msg=msg_error)

    def test_angle_difference(self):
        msg_error = "Angle difference is incorrect"
        angle1 = (180+30 + 10)*np.pi/180
        angle2 = (10)*np.pi/180
        self.assertAlmostEqual(pfilter.angle_abs_pi(angle1,angle2)*180/np.pi,-180+30,8,msg = msg_error)

        angle1 = -(180 + 30 + 10) * np.pi / 180
        angle2 = -(10) * np.pi / 180
        self.assertAlmostEqual(pfilter.angle_abs_pi(angle1, angle2) * 180 / np.pi, 180 - 30, 8, msg=msg_error)

        angle1 = (270 + 30 + 10) * np.pi / 180
        angle2 = (10) * np.pi / 180
        self.assertAlmostEqual(pfilter.angle_abs_pi(angle1, angle2) * 180 / np.pi, -60 , 8, msg=msg_error)

        angle1 = (270 + 30 + 360 + 10) * np.pi / 180
        angle2 = (10) * np.pi / 180
        self.assertAlmostEqual(pfilter.angle_abs_pi(angle1, angle2) * 180 / np.pi, -60, 8, msg=msg_error)

    def test_landmark_model_correspondence(self):
        xt = self.xodot1 + np.array((0.01, 0.01, 3 * np.pi / 180))

        landmark_std = [0.1, 0.1]
        odom_std = [0.1, 0.01, 0.02, 0.01]

        # Landmarks positons map
        m = np.stack((np.concatenate((np.arange(5), np.arange(5))),
                      np.concatenate((np.ones(5, dtype=float) * 2, np.ones(5, dtype=float) * 3)),
                      np.linspace(0, 9, 10)))  # map of environment (landmarks)
        xreal = pfilter.sample_motion_model_odometry(self.ut, self.x0, odom_std)
        xreal.resize(3)

        # Computes landmarks measures
        zt = np.stack(
            (np.sqrt((m[0, :] - xreal[0]) ** 2 + (m[1, :] - xreal[1]) ** 2) + landmark_std[0] * np.random.randn(),
             np.arctan2(m[1, :] - xreal[1], m[0, :] - xreal[0]) - xreal[2] + landmark_std[1] * np.random.randn(),
             m[2, :]))

        fi = np.copy(zt[0:2, :])
        ci = np.copy(zt[2, :])
        msg_error = "Probability value must be greater than zero"
        self.assertGreaterEqual(pfilter.landmark_model_correspondence(fi, ci, xt, m, landmark_std, N_particles=1),0,msg=msg_error)

        msg_error = "If there are no measures, landmark_model_correspondence() function must return None"
        self.assertIsNone(pfilter.landmark_model_correspondence(None, ci, xt, m, landmark_std, N_particles=1),msg=msg_error)


if __name__ == '__main__':
    unittest.main()
