import numpy as np
import torch

from envs.env_base import BaseEnv

## STATE
# x = [p_x, p_y, theta]
## CONTROL
# u = [v, w]

# Denote angle indices to handle smooth transition
ANGLE_IDX = [2]

# TURTLEBOT PARAMETERS
# k1, k2, k3 = 0.9061, 0.8831, 0.8548
k1, k2, k3 = 1.0, 1.0, 1.0

# X bounds
X_MIN = np.array([-10.0, -10.0, 0]).reshape(-1, 1)
X_MAX = np.array([10.0, 10.0, 2 * np.pi]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-1.0, -1.0, (1 / 2) * np.pi])
XREF_INIT_MAX = np.array([1.0, 1.0, (3 / 2) * np.pi])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.array([-0.5, -0.5, -(1 / 4) * np.pi])
XE_INIT_MAX = np.array([0.5, 0.5, (1 / 4) * np.pi])

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([0.0, -1.82]).reshape(-1, 1)
UREF_MAX = np.array([0.22, 1.82]).reshape(-1, 1)

env_config = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 3,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.05,
    "time_bound": 30.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.0,  # control cost weight
}


class TurtlebotEnv(BaseEnv):
    def __init__(
        self,
        sample_mode: str = "uniform",
        reward_mode: str = "default",
    ) -> None:
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "turtlebot"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["reward_mode"] = reward_mode

        super(TurtlebotEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]
        f = lib.zeros((n, self.num_dim_x))
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]

        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 0, 0] = k1 * lib.cos(theta)
        B[:, 1, 0] = k2 * lib.sin(theta)
        B[:, 2, 1] = k3
        return B

    def _B_null_logic(self, x, n, lib):
        """
        Calculates the orthogonal complement B_null(x) (or B_bot).
        This logic is taken from your 'Bbot_func'.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]

        Bbot = lib.zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control))

        Bbot[:, 0, 0] = k2 * lib.sin(theta) * k3
        Bbot[:, 1, 0] = -k1 * lib.cos(theta) * k3
        Bbot[:, 2, 0] = 0.0

        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        linear_velocity = UREF_MAX[0] * np.random.uniform(0.2, 0.8)
        uref = np.array([linear_velocity.squeeze(), 0])
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    0.0,
                    weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)

        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref

    def system_reset(self):
        """Resets the system to an initial state and generates a reference trajectory."""
        xref_0, xe_0, x_0 = self.define_initial_state()

        # Generate reference trajectory
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()

        xref_list, xref_wrapped_list, uref_list = [xref_0], [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(
                freqs, weights, _t, {"xref_0": xref_0}
            )
            xref_t, xref_wrapped_t, term, trunc, _ = self.get_transition(
                xref_list[-1].copy(), uref_t
            )

            xref_list.append(xref_t)
            xref_wrapped_list.append(xref_wrapped_t)
            uref_list.append(uref_t)

            if term or trunc:
                break

        return (
            x_0,
            np.array(xref_wrapped_list),
            np.array(uref_list),
            i + 1,
        )
