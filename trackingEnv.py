import copy
import gym
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random


def reward_tracking_vision(x, y, z,
                           v=np.zeros((4,)),
                           u=np.zeros((4,)),
                           optimal_distance=2.0,
                           max_dist=15.0,
                           min_dist=1.0,
                           exp=(1 / 3),
                           alpha=0.0,
                           beta=0.0,
                           max_steps=400.0):
    done = False

    y_ang = np.arctan(y / x)
    z_ang = np.arctan(z / x)

    y_error = abs(y_ang / (np.pi / 4))
    z_error = abs(z_ang / (np.pi / 4))
    x_error = abs(x - optimal_distance)

    z_rew = max(0, 1 - z_error)
    y_rew = max(0, 1 - y_error)
    x_rew = max(0, 1 - x_error)

    vel_penalty = np.linalg.norm(v) / (1 + np.linalg.norm(v))
    u_penalty = np.linalg.norm(u) / (1 + np.linalg.norm(u))

    reward_track = (x_rew * y_rew * z_rew) ** exp

    reward = (reward_track - alpha * vel_penalty - beta * u_penalty) * (400 / max_steps)

    if abs(np.linalg.norm(np.array([x, y, z]))) > max_dist or abs(np.linalg.norm(np.array([x, y, z]))) < min_dist:
        done = True
        reward = -10 / (400 / max_steps)

    return reward, done


def drone_dyn(X, t, g, m, w, f):
    X = np.expand_dims(X, axis=1)

    # Variables and Parameters
    zeta = np.array([0, 0, 1]).reshape(3, 1)
    gv = np.array([0, 0, -1]).reshape(3, 1) * g
    p = X[0: 3, 0]
    v = X[3: 6, 0]
    R = X[6: 15].reshape(3, 3)

    # Drone Dynamics
    dp = v
    dv = np.dot(R, zeta) * f / m + gv
    dR = np.dot(R, sKw(w))

    # Output
    dX = np.concatenate((dp.reshape(-1, 1), dv, dR.reshape(-1, 1)), axis=0).squeeze()
    return dX


def sKw(x):
    Y = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)
    return Y


class TrackingEnv(gym.Env):
    def __init__(self,
                 optimal_distance=0.75,
                 Ts=0.05,
                 sensor_noise=False,
                 mass_noise=False,
                 start_noise=False,
                 output_delay=False,
                 move_target=True,
                 asymmetric_actor_buffer_length=15):
        self.action_limit = 4
        self.action_space = gym.spaces.Box(-self.action_limit, self.action_limit, shape=(4,), dtype=np.float32)
        self.asymmetric_actor_buffer_length = asymmetric_actor_buffer_length
        print("df_env")
        # Observation Space
        critic_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        actor_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.asymmetric_actor_buffer_length * 3,),
                                         dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'actor': actor_obs_space,
            'critic': critic_obs_space
        })

        self.optimal_distance = optimal_distance  # 2.0 # 0.75

        # NOISE VARIABLES AND PARAMETERS
        self.sensor_noise = sensor_noise
        self.mass_noise = mass_noise
        self.start_noise = start_noise
        self.output_delay = output_delay

        # Output Delay
        self.max_delay = 1
        if self.output_delay:
            self.delay = random.randint(0, self.max_delay)
        else:
            self.delay = 0
        self.delayed_output = None

        # Start Noise
        self.start_noise_var = 0.10

        # Sensor Noise
        self.sensor_noise_var = 0.003

        # Mass Noise
        self.a = 0.4  # 0.6  # 0.4
        self.b = 1.6  # 1.4  # 1.6
        if self.mass_noise:
            self.m = random.uniform(self.a, self.b) # # Tracker Mass[kg]
        else:
            self.m = 1  # # Tracker Mass[kg]

        # RL TRAINING
        # Variables
        self.state = None
        self.reward = 0
        self.done = 0
        self.episode_steps = 0
        self.info = {"tracker_position": None,
                     "tracker_rotation": None,
                     "target_position": None,
                     "target_rotation": None,
                     "jerk": None,
                     "w": None}

        # Actor History
        self.actor_state = None

        # SIMULATION PARAMETERS
        self.g = 9.8  # Gravitational Acceleration[m / s ^ 2]
        self.Tin = 0  # Initial time[s]
        self.Ts = Ts  # Sampling time[s] 0.05

        # EPISODE
        self.episode_time = 40  # s
        self.max_episode_steps = int(self.episode_time / self.Ts)
        self.stop_step = random.randint(0, self.max_episode_steps)
        self.stop_duration = 10 / self.Ts

        # DRONE TRACKER PARAMETERS
        self.Xin = None

        # TARGET MOVEMENT
        self.T_target = 0.0
        self.Ts_target = self.Ts
        self.move_target = move_target
        self.reset_target()
        self.a1, self.a2, self.a3 = None, None, None
        self.phi1, self.phi2, self.phi3 = None, None, None
        self.ws1, self.ws2, self.ws3 = None, None, None
        self.pr0 = np.array([0.0, 0.0, 0.0])
        self.prout = self.pr0
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.eye(3)

        # RENDER
        self.fig = None
        self.ax = None
        self.axs = []

        self.position_x = []
        self.position_y = []
        self.position_z = []


    def reset_target(self, p=None, rand=False):
        self.a1 = 1 + random.random() * 30
        self.a2 = 1 + random.random() * 30
        self.a3 = 1 + random.random() * 3

        k1 = 5 + random.random() * 10
        k2 = 5 + random.random() * 10
        k3 = 5 + random.random() * 10

        self.phi1 = -np.pi / 2 + random.random() * np.pi / 2
        self.phi2 = -np.pi / 2 + random.random() * np.pi / 2
        self.phi3 = -np.pi / 2 + random.random() * np.pi / 2

        self.ws1 = 2 * np.pi / self.a1 / k1
        self.ws2 = 2 * np.pi / self.a2 / k2
        self.ws3 = 2 * np.pi / self.a3 / k3

        if p is not None:
            self.pr0 = np.array([p[0] + self.optimal_distance, p[1], p[2]])
            if rand:
                check = False
                while not check:
                    xdis = np.random.randn() * self.start_noise_var + self.optimal_distance
                    ydis = np.random.randn() * self.start_noise_var
                    zdis = np.random.randn() * self.start_noise_var

                    self.pr0 = np.array([p[0] + xdis, p[1] + ydis, p[2] + zdis])

                    if abs(np.linalg.norm(np.array([xdis, ydis, zdis]))) > 0.4 and abs(
                            np.linalg.norm(np.array([xdis, ydis, zdis]))) < 10.0:
                        check = True
        else:
            self.pr0 = np.array([0.0, 0.0, 0.0])

        self.prout = self.pr0
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.eye(3)

    def step(self, u):
        inp = u.reshape((self.action_space.shape[0],))

        # Update Info
        self.info["tracker_position"] = self.Xin[0: 3].reshape(3, )
        self.info["tracker_rotation"] = self.Xin[6: 15].reshape(3, 3)
        self.info["target_position"] = self.prout.reshape(3, )
        self.info["target_rotation"] = self.Rrout.reshape(3, 3)
        self.info["jerk"] = inp[:-1].reshape(3, )
        self.info["w"] = np.array(inp[-1]).reshape(1, )
        # print(self.info)

        self.episode_steps += 1

        # Target movement
        if self.move_target and (
                self.episode_steps <= self.stop_step or self.episode_steps > (self.stop_step + self.stop_duration)):
            self.prout = np.array(
                [self.a1 * np.sin(self.phi1 + self.ws1 * self.T_target) - self.a1 * np.sin(self.phi1) + self.pr0[0],
                 self.a2 * np.sin(self.phi2 + self.ws2 * self.T_target) - self.a2 * np.sin(self.phi2) + self.pr0[1],
                 self.a3 * np.sin(self.phi3 + self.ws3 * self.T_target) - self.a3 * np.sin(self.phi3) + self.pr0[2]])
            self.vrout = np.array(
                [self.a1 * self.ws1 * np.cos(self.phi1 + self.T_target * self.ws1),
                 self.a2 * self.ws2 * np.cos(self.phi2 + self.T_target * self.ws2),
                 self.a3 * self.ws3 * np.cos(self.phi3 + self.T_target * self.ws3)])
            self.arout = np.array(
                [-self.a1 * self.ws1 ** 2 * np.sin(self.phi1 + self.T_target * self.ws1),
                 -self.a2 * self.ws2 ** 2 * np.sin(self.phi2 + self.T_target * self.ws2),
                 -self.a3 * self.ws3 ** 2 * np.sin(self.phi3 + self.T_target * self.ws3)])

            self.T_target += self.Ts_target
        else:
            self.vrout = np.array([0, 0, 0])
            self.arout = np.array([0, 0, 0])

        # INTEGRATE TRACKER DYNAMICS
        t = [self.Tin, self.Tin + self.Ts]
        u = u.reshape(self.action_space.shape[0],)

        w = u[:3]
        df = u[3] * 5

        # Delay Output
        if self.delayed_output is None:
            self.delayed_output = []
            self.delayed_output = [{"w": copy.deepcopy(w), "f": copy.deepcopy(self.fk)}] * (self.max_delay + 1)

        self.delayed_output.insert(0, {"w": copy.deepcopy(w), "f": copy.deepcopy(self.fk)})
        self.delayed_output.pop()
        current_output = self.delayed_output[self.delay]

        # Conpute Dynamics
        Xout = odeint(drone_dyn, self.Xin.squeeze(), t, args=(self.g, self.m, current_output["w"], current_output["f"]))  # X, t, g, m, w, f

        # Clip Force -> Actuators constraint
        self.fk = np.clip(self.fk + df * self.Ts, 0.1, self.g * 2)

        Xout = Xout[-1, :].T
        Tout = t[-1]

        # Tracker output variables
        pout = Xout[0: 3]
        vout = Xout[3: 6]
        Rout = Xout[6: 15].reshape(3, 3)

        zeta = np.array([0, 0, 1]).reshape(3, 1)
        gv = np.array([0, 0, -1]).reshape(3, 1) * self.g
        aout = (np.dot(Rout, zeta) * (current_output["f"] / self.m) + gv).reshape(3, )

        # X Y Z of the target wrt to the tracker body frame at time Tout
        x, y, z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.prout - pout)

        v_x, v_y, v_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.vrout - vout)

        a_x, a_y, a_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.arout - aout)

        base_state = np.array([x - self.optimal_distance, y, z, v_x, v_y, v_z, a_x, a_y, a_z]).reshape((1, 9))
        state_actor = np.array([x - self.optimal_distance, y, z]).reshape((1, 3))

        # SENSOR NOISE
        if self.sensor_noise:
            noise_x = np.random.randn() * self.sensor_noise_var
            noise_y = np.random.randn() * self.sensor_noise_var
            noise_z = np.random.randn() * self.sensor_noise_var
            noise_vector = np.array([noise_x,
                                     noise_y,
                                     noise_z]).reshape((1, 3))
            actor_obs = state_actor + noise_vector
        else:
            actor_obs = state_actor

        # ACTOR HISTORY
        if self.actor_state is None:
            self.actor_state = []
            self.actor_state = [actor_obs] * self.asymmetric_actor_buffer_length

        self.actor_state.append(actor_obs)
        self.actor_state.pop(0)

        current_state_actor = np.concatenate(self.actor_state, axis=1)

        self.state = {
            'actor': current_state_actor,
            'critic': base_state
        }

        # REWARD AND DONE
        self.reward, self.done = reward_tracking_vision(x, y, z,
                                                        v=np.array([v_x, v_y, v_z]),
                                                        u=u / np.array([self.action_limit, self.action_limit, self.action_limit, self.action_limit * 5]),
                                                        optimal_distance=self.optimal_distance,
                                                        min_dist=0.4,
                                                        max_dist=10.0,
                                                        alpha=0.4,
                                                        beta=0.4,
                                                        max_steps=self.max_episode_steps)

        # Update loop states
        self.Xin = Xout
        self.Tin = Tout

        if self.episode_steps >= self.max_episode_steps:
            self.done = True

        return self.state, self.reward, self.done, self.info

    def reset(self):
        # RL AND ACTOR VARIABLES
        self.episode_steps = 0
        self.actor_state = None

        # NOISE AND RANDOMIZATION
        self.stop_step = random.randint(0, self.max_episode_steps)
        if self.mass_noise:
            self.m = random.uniform(self.a, self.b)  # # Tracker Mass[kg]

        self.delayed_output = None
        if self.output_delay:
            self.delay = random.randint(0, self.max_delay)
        else:
            self.delay = 0

        # RENDERING
        self.position_x = []
        self.position_y = []
        self.position_z = []

        # INITIAL CONDITIONS
        # Tracker
        pin = np.random.uniform(-2, 2, (3, 1))  # Tracker Initial Position [m]
        vin = np.array([0, 0, 0]).reshape(3, 1)  # Tracker Initial Velocity [m/s]
        Rin = np.eye(3)  # Tracker Initial Attitude (Body -> Inertial)
        self.fk = self.g
        self.Xin = np.concatenate((pin, vin, Rin.reshape(-1, 1)), axis=0)  # Tracker State
        self.Tin = 0

        # Target
        self.T_target = 0
        self.reset_target(pin.reshape((3,)), rand=self.start_noise)

        u = np.zeros((self.action_space.shape[0]), dtype=np.float64)

        self.step(u)
        self.done = False

        return self.state

    def render(self, mode='human', critic=0):
        if mode == 'human':
            if self.fig is None:
                self.fig = plt.figure()
                self.ax = plt.axes(projection='3d')
            p_tracker = self.Xin[0: 3]
            b_tracker = self.Xin[6: 15].reshape(3, 3)[:, 0].reshape(3, )
            p_target = self.prout
            self.position_x.append(p_tracker[0])
            self.position_y.append(p_tracker[1])
            self.position_z.append(p_tracker[2])
            self.ax.cla()
            self.ax.plot3D(self.position_x, self.position_y, self.position_z, 'gray')
            self.ax.scatter3D(p_target[0], p_target[1], p_target[2], 'red')
            self.ax.scatter3D(p_tracker[0], p_tracker[1], p_tracker[2], 'red')
            self.ax.quiver(p_tracker[0], p_tracker[1], p_tracker[2], b_tracker[0], b_tracker[1], b_tracker[2],
                           length=0.2, normalize=False, color='g')
            plt.pause(0.00001)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)