# View local Isaac Sim from a Mac

The Linux renderer is `minjae-uiuc` (`128.174.245.104`), using Isaac Sim 5.1 and the RTX 4070 SUPER. Use NVIDIA's Isaac Sim WebRTC Streaming Client on the Mac:

- [Apple Silicon download](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-macos-arm64.dmg)
- [Intel download](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-macos-x64.dmg)

Open the DMG and drag the app into Applications. Keep the Mac on the campus network or UIUC VPN, then connect the app to `128.174.245.104` using the default signaling port, `49100`.

## Network access

The server uses TCP 49100 for signaling and UDP 47998 for video. Ordinary SSH port forwarding alone does not carry the UDP video stream.

UFW is enabled on Linux. The current session cannot inspect or change its rules because sudo requires a password, but it can read the firewall log. The log confirms the Mac's new streaming connections arrive from `10.250.133.62`; the address in the existing SSH session was `10.192.157.104`. On the Linux machine, allow these ports from the source address seen in the firewall log:

```bash
sudo ufw allow from 10.250.133.62 to 128.174.245.104 port 49100 proto tcp
sudo ufw allow from 10.250.133.62 to 128.174.245.104 port 47998 proto udp
```

If the Mac's network or VPN address changes, check the source of new connections in `/var/log/ufw.log`. An older SSH session's address can differ from the source of new connections. The user applied the corrected rules above and confirmed the Mac's TCP connection test succeeds, the scene is visible, and camera controls work.

From a Mac terminal, check signaling reachability:

```bash
nc -vz 128.174.245.104 49100
```

Seeing the scene in the streaming client verifies the video connection; a successful TCP check alone does not.

## Current viewer and experiment rendering

The initial connection test used Isaac Lab's installed `scripts/tutorials/00_sim/launch_app.py` demo. It has been replaced with continuous C2RL-PPO car playback using the completed 500,000-step checkpoint from sweep run `wncagd2n`. Its command, checkpoint, configuration, PID, and log are recorded under `experiments/local_rendering/20261019T204142Z-c2rl-ppo-car/`. The simulation uses analytical car dynamics rendered in Isaac: the blue body is the car and the green sphere is its reference. Playback uses the deterministic policy mean, without training, and continues across episode resets.

Only one streaming Isaac instance can use these ports at a time. Before replacing the simulation, stop the current viewer using its recorded PID. For the existing Isaac training or playback entrypoint, set `PUBLIC_IP=128.174.245.104` and add `--livestream 1` plus `--kit_args '--/app/livestream/fixedHostPort=47998 --/exts/omni.services.transport.server.http/host=127.0.0.1'`. Add `--viewer` to `scripts/skrl/play.py` for continuous real-time playback; without this flag, playback remains the normal finite evaluator. The analytical car task is `Isaac-Classic-Car-v0`; Isaac renders its analytical plant states. Existing manifests retain the task names used when those viewers started. The `--classic` route does not launch Isaac Sim.

Car playback now adds a dark slate floor with a one-meter grid, a green line for the full episode reference, and an orange line for the car's traveled path. The blue marker remains the car. Both paths refresh on episode reset, and the initial camera frames the reference and initial car position. This display lives in `scripts/skrl/car_viewer.py` and is enabled only by playback's `--viewer` route. The current run is recorded under `experiments/local_rendering/20261019T205526Z-car-platform-paths/`. After a viewer restart, fully quit the Mac client with Command-Q, reopen it, and reconnect to the same server.

For a larger initial position gap, add `--viewer_initial_offset 0 3`: each analytical-car episode starts with the car 3 m along world y from the reference, while retaining the usual heading and speed perturbations. This affects only the viewer's reset distribution. Playback includes bounded, held disturbances by default: heading-rate and acceleration scales are 0.02 rad/s and 0.05 m/s², with no direct position noise. Add `--disable_disturbance` to turn them off. Episode-start offsets, gap and disturbance samples are printed in the viewer log.

The previous 3 m-gap playback is recorded under `experiments/local_rendering/20261019T210454Z-car-initial-gap-3m/`.

The previous rectangular-offset playback is recorded under `experiments/local_rendering/20261019T211421Z-car-offset-box-gusts/`.

The active playback uses `--viewer_initial_offset_signed_range 5 10 --viewer_disturbance_bounds 0.1 0.2`: each of the x and y initial position errors independently samples a magnitude uniformly in [5,10] m and an equally likely positive or negative sign. Each coordinate therefore samples uniformly from the union [-10,-5] and [5,10] m, with equal probability for the two intervals. Heading-rate and acceleration gust bounds are 0.1 rad/s and 0.2 m/s². Its records are under `experiments/local_rendering/20261019T211905Z-car-independent-signed-offsets/`; prior viewers were stopped and their records retained.

[NVIDIA's Isaac Sim 5.1 streaming instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html)
