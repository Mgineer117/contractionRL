"""Body-frame actuators for the simplified PhysX rigid bodies."""
import torch


def quadrotor_quat(angles):
    """Classic attitude to wxyz quaternion: R = Rx(-roll) Ry(-pitch) Rz(yaw)."""
    roll, pitch, yaw = (angles / 2).unbind(-1)
    cr, cp, cy = roll.cos(), pitch.cos(), yaw.cos()
    sr, sp, sy = roll.sin(), pitch.sin(), yaw.sin()
    return torch.stack((cr*cp*cy - sr*sp*sy, -sr*cp*cy - cr*sp*sy,
                        sr*cp*sy - cr*sp*cy, cr*cp*sy + sr*sp*cy), -1)


def quadrotor_angles(q):
    """Inverse of quadrotor_quat within the shared tilt domain (±pi/3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((torch.atan2(2*(y*z-w*x), 1-2*(x*x+y*y)),
                        torch.asin((-2*(x*z+w*y)).clamp(-1., 1.)),
                        torch.atan2(2*(w*z-x*y), 1-2*(y*y+z*z))), -1)


def commanded_body_rates(x, u, quad):
    """Map the analytical rate controls to body angular velocity, without lag."""
    if not quad:
        rates = x.new_zeros(len(x), 3)
        rates[:, 2] = u[:, 0]
        return rates
    pitch, yaw = x[:, 8], x[:, 9]
    roll_rate, pitch_rate, yaw_rate = u[:, 1:4].unbind(-1)
    return torch.stack((-roll_rate*pitch.cos()*yaw.cos() - pitch_rate*yaw.sin(),
                        roll_rate*pitch.cos()*yaw.sin() - pitch_rate*yaw.cos(),
                        yaw_rate + roll_rate*pitch.sin()), -1)


def body_wrench(x, u, quad):
    """Mass is 1 kg; rates are imposed separately, so there is no torque servo."""
    force, torque = x.new_zeros(len(x), 3), x.new_zeros(len(x), 3)
    if quad:
        force[:, 2] = x[:, 6]
    else:
        force[:, 0] = u[:, 1]
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
