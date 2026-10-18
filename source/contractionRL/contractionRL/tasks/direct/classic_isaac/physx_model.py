"""Domains and body-frame actuators for the simplified PhysX rigid bodies."""
import math
import numpy as np
import torch


def domain(task):
    if task == 'classic-quadrotor-v0':
        low = [-30., -30., .15, -1.5, -1.5, -1.5, 4.905] + [-math.pi/3] * 3 + [-10.] * 3
        high = [30., 30., 30., 1.5, 1.5, 1.5, 19.62] + [math.pi/3] * 3 + [10.] * 3
        controls = [60., 10., 10., 10.]
        names = ('pos_x', 'pos_y', 'pos_z', 'vel_x_w', 'vel_y_w', 'vel_z_w', 'thrust',
                 'roll', 'pitch', 'yaw', 'ang_vel_x_b', 'ang_vel_y_b', 'ang_vel_z_b')
        angles = [7, 8, 9]
        gust = [0.] * 3 + [.2] * 3 + [0.] * 4 + [.05] * 3
    elif task in ('classic-cartpole-v0', 'classic-segway-v0'):
        high = [5., math.pi/3 if task == 'classic-cartpole-v0' else .90, 1., 1. if task == 'classic-cartpole-v0' else math.pi]
        low = [-value for value in high]
        controls = [48.]
        names = ('pos_x', 'pitch', 'vel_x_b', 'pitch_rate')
        angles = [1]
        gust = [0., 0., .05, .05]
    elif task in ('classic-car-v0', 'classic-car-v1'):

        low = [-45., -45., -math.pi, .2 if task.endswith('v1') else 1., -2., -4.]
        high = [45., 45., math.pi, 2., 2., 4.]
        controls = [3., 3.]
        names = ('pos_x', 'pos_y', 'yaw', 'vel_x_b', 'vel_y_b', 'ang_vel_z_b')
        angles = [2]
        gust = [0., 0., 0., .05, .05, .02]
    else:
        raise ValueError(f'No PhysX model for {task}')
    low, high = np.array(low, np.float32), np.array(high, np.float32)
    margin = .05 * (high - low)
    if len(low) == 6:
        margin[2] = 0.
    return low, high, np.array(controls, np.float32), names, angles, np.array(gust, np.float32), margin


def body_wrench(x, u, inertia, quad):
    """Mass is 1 kg; yaw/body rates are servo targets, not assigned velocities.

    Cars use lateral viscous drag (6 /s) and yaw-rate feedback (12 /s).
    Quads use mass-normalized thrust and body-rate feedback (12 /s).
    Torque saturation and finite inertia produce actual rate-tracking lag.
    """
    force, torque = x.new_zeros(len(x), 3), x.new_zeros(len(x), 3)
    if quad:
        force[:, 2] = x[:, 6]
        omega = x[:, 10:13]
        torque = (12 * (u[:, 1:4] - omega) * inertia + torch.linalg.cross(omega, omega * inertia)).clamp(-.2, .2)
    else:
        force[:, 0] = u[:, 1]
        force[:, 1] = -6 * x[:, 4]
        torque[:, 2] = (12 * (u[:, 0] - x[:, 5]) * inertia[2]).clamp(-1., 1.)
    return force, torque


def segway_parameters():
    """Match the classic upright gravity/input linearization, not its full nonlinear equations."""
    mass_total = 208.3 / 9.8
    gravity_slope = 208.3 / 23.7
    hinge_inertia = 9.81 / gravity_slope + 1 / mass_total
    bv, bw = 12.7 / 23.7, -47.9 / 23.7
    torque_gain = -(bv + hinge_inertia * bw)
    radius = torque_gain / (mass_total * bv + bw)
    return mass_total-1, hinge_inertia-1, torque_gain, radius, 79.9/12.7
