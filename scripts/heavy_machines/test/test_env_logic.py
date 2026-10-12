"""Runs HeavyMachineEnv's real control code offline (needs torch, not Isaac Sim).

Isaac Lab modules are replaced by minimal stubs; a fake articulation records
what the environment writes.  Skipped automatically when torch is missing.

    python scripts/heavy_machines/test/test_env_logic.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _install_isaac_stubs():
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "_hm_stub", False):
        return

    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def mod(name):
        m = types.ModuleType(name)
        m._hm_stub = True
        sys.modules[name] = m
        return m

    isaaclab = mod("isaaclab")
    sim = mod("isaaclab.sim")
    conv = mod("isaaclab.sim.converters")
    spawners = mod("isaaclab.sim.spawners")
    from_files = mod("isaaclab.sim.spawners.from_files")
    act = mod("isaaclab.actuators")
    assets = mod("isaaclab.assets")
    envs = mod("isaaclab.envs")
    scene = mod("isaaclab.scene")
    sensors = mod("isaaclab.sensors")
    utils = mod("isaaclab.utils")
    isaaclab.sim, isaaclab.actuators, isaaclab.assets = sim, act, assets
    isaaclab.envs, isaaclab.scene, isaaclab.sensors, isaaclab.utils = envs, scene, sensors, utils
    sim.converters, sim.spawners, spawners.from_files = conv, spawners, from_files

    for n in ("SimulationCfg", "DomeLightCfg", "UrdfFileCfg", "RigidBodyPropertiesCfg",
              "ArticulationRootPropertiesCfg", "PinholeCameraCfg", "CollisionPropertiesCfg",
              "PreviewSurfaceCfg", "ConeCfg", "SphereCfg", "CuboidCfg", "CylinderCfg"):
        setattr(sim, n, type(n, (_Cfg,), {}))
    from_files.GroundPlaneCfg = type("GroundPlaneCfg", (_Cfg,), {})
    from_files.spawn_ground_plane = lambda *a, **k: None

    class UrdfConverterCfg(_Cfg):
        class JointDriveCfg(_Cfg):
            class PDGainsCfg(_Cfg):
                pass
    conv.UrdfConverterCfg = UrdfConverterCfg
    act.ImplicitActuatorCfg = type("ImplicitActuatorCfg", (_Cfg,), {})

    class ArticulationCfg(_Cfg):
        class InitialStateCfg(_Cfg):
            pass
    assets.ArticulationCfg = ArticulationCfg
    assets.Articulation = object

    class DirectRLEnv:
        def _reset_idx(self, env_ids):
            pass
    envs.DirectRLEnv = DirectRLEnv
    envs.DirectRLEnvCfg = type("DirectRLEnvCfg", (_Cfg,), {})
    scene.InteractiveSceneCfg = type("InteractiveSceneCfg", (_Cfg,), {})

    class CameraCfg(_Cfg):
        class OffsetCfg(_Cfg):
            pass
    sensors.CameraCfg = CameraCfg
    sensors.Camera = object
    utils.configclass = lambda cls: cls


class FakeRobot:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.root_pose = None
        self.joint_pos = None

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.root_pose = pose.clone()

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.root_vel = vel.clone()

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        assert pos.shape == vel.shape
        self.joint_pos = pos.clone()

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.target = target.clone()


class FakeCam:
    def set_world_poses(self, positions, orientations, env_ids=None, convention="ros"):
        assert convention == "world"
        self.positions, self.orientations = positions.clone(), orientations.clone()


def make_env(name, num_envs=2, cameras=True, seed=0):
    from heavy_machines import kinematics as K
    from heavy_machines import registry
    from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg

    spec = registry.get(name)
    env = HeavyMachineEnv.__new__(HeavyMachineEnv)
    env.cfg = make_env_cfg(spec.name, enable_cameras=cameras, status_interval=0)
    env.machine_spec = spec
    env._root_height = K.root_height(spec)
    env._spawn_rng = random.Random(seed)
    env.device = "cpu"
    env.num_envs = num_envs
    names = [j.name for j in spec.movable_joints]
    random.Random(seed).shuffle(names)            # Isaac's joint order differs from the URDF's
    env.robot = FakeRobot(names)
    env.scene = types.SimpleNamespace(env_origins=torch.tensor([[100.0 * i, 0.0, 0.0] for i in range(num_envs)]))
    env.cams = {c.name: FakeCam() for c in spec.cameras} if cameras else {}
    env.last_spawn = [{} for _ in range(num_envs)]
    env._build_joint_maps()
    env._build_buffers()
    env._substep = 0
    env._control_step = 0
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    env.max_episode_length = 1000
    return env, spec


def run(env, action, steps):
    """Step the control code; returns the accumulated (unwrapped) yaw change per env."""
    a = torch.tensor([action] * env.num_envs, dtype=torch.float32)
    turned = torch.zeros(env.num_envs)
    for _ in range(steps):
        env._pre_physics_step(a)
        for _ in range(env.cfg.decimation):
            before = env._yaw.clone()
            env._apply_action()
            turned += torch.atan2(torch.sin(env._yaw - before), torch.cos(env._yaw - before))
    return turned


def steps_to_sweep(spec) -> int:
    """Control steps for the slowest tool to cross its whole range, plus margin."""
    worst = 2.0
    for t in spec.tools:
        j = spec.joint(t.joint)
        span = math.pi if j.type == "continuous" else j.upper - j.lower
        worst = max(worst, span / j.velocity)
    return int(30 * (worst * 1.2 + 1.0))


@unittest.skipIf(torch is None, "torch not installed")
class TestEnvLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_isaac_stubs()
        from heavy_machines import registry
        cls.names = registry.names()

    def test_tools_respect_limits_and_wrap(self):
        for name in self.names:
            with self.subTest(machine=name):
                env, spec = make_env(name)
                for sign in (1.0, -1.0):
                    run(env, [0.0, 0.0] + [sign] * len(spec.tools), steps_to_sweep(spec))
                    for t in spec.tools:
                        j = spec.joint(t.joint)
                        q = env._q[:, env._jidx[j.name]]
                        if j.type == "continuous":
                            self.assertTrue(bool((q.abs() <= math.pi + 1e-5).all()))
                        else:
                            target = j.upper if sign > 0 else j.lower
                            self.assertTrue(torch.allclose(q, torch.full_like(q, target), atol=1e-4),
                                            f"{t.label} should reach {target}, got {q.tolist()}")
                self.assertTrue(bool(torch.isfinite(env.robot.joint_pos).all()))

    def test_drive_forward_moves_along_heading(self):
        for name in self.names:
            with self.subTest(machine=name):
                env, spec = make_env(name)
                env._yaw[:] = 0.7
                run(env, [1.0, 0.0] + [0.0] * len(spec.tools), 60)
                dist = 2.0 * spec.drive.max_speed
                self.assertAlmostEqual(env._xy[0, 0].item(), dist * math.cos(0.7), delta=0.02 * dist)
                self.assertAlmostEqual(env._xy[0, 1].item(), dist * math.sin(0.7), delta=0.02 * dist)
                self.assertAlmostEqual(env._yaw[0].item(), 0.7, places=4)

    def test_turn_left_increases_yaw(self):
        for name in self.names:
            with self.subTest(machine=name):
                env, spec = make_env(name)
                turned = run(env, [0.7, 1.0] + [0.0] * len(spec.tools), 90)
                self.assertGreater(turned[0].item(), 0.05, "turn=+1 must turn left")
                env2, _ = make_env(name)
                turned = run(env2, [0.7, -1.0] + [0.0] * len(spec.tools), 90)
                self.assertLess(turned[0].item(), -0.05, "turn=-1 must turn right")

    def test_lateral_acceleration_limited(self):
        """Wheeled machines at full speed and full lock stay under max_lat_accel (after settling)."""
        for name in self.names:
            spec = make_env(name)[1]
            if spec.drive.type == "skid":
                continue
            with self.subTest(machine=name):
                env, _ = make_env(name)
                run(env, [1.0, 1.0] + [0.0] * len(spec.tools), 300)
                lat = (env._speed * env._yaw_rate).abs().max().item()
                self.assertLessEqual(lat, spec.drive.max_lat_accel * 1.05 + 1e-6)

    def test_steering_and_articulation_joints(self):
        for name in self.names:
            spec = make_env(name)[1]
            d = spec.drive
            if d.type == "skid":
                continue
            with self.subTest(machine=name):
                env, _ = make_env(name)
                run(env, [0.3, 1.0] + [0.0] * len(spec.tools), 90)
                if d.type == "ackermann":
                    angles = env._q[0, env._steer_ids]
                    ys = env._steer_y
                    left, right = angles[ys.argmax()].item(), angles[ys.argmin()].item()
                    sign = 1.0 if d.steer_axle == "front" else -1.0
                    self.assertGreater(sign * left, 0.0, "steered wheels must point into the turn")
                    self.assertGreater(abs(left), abs(right), "inner (left) wheel must steer more")
                else:
                    g = env._q[0, env._artic_idx].item()
                    self.assertLess(g, 0.0, "turning left must give a negative articulation angle")
                    aj = spec.joint(d.articulation_joint)
                    self.assertGreaterEqual(g, aj.lower - 1e-6)

    def test_skid_turn_in_place(self):
        for name in self.names:
            env, spec = make_env(name)
            if spec.drive.type != "skid":
                continue
            with self.subTest(machine=name):
                run(env, [0.0, 1.0] + [0.0] * len(spec.tools), 30)
                self.assertAlmostEqual(env._yaw[0].item(), spec.drive.max_yaw_rate, delta=1e-3)
                self.assertLess(env._xy[0].abs().max().item(), 1e-6)

    def test_reset_and_cameras(self):
        for name in self.names:
            with self.subTest(machine=name):
                env, spec = make_env(name)
                env.cfg.spawn_pose = (3.0, -2.0, 1.2)
                run(env, [1.0, 1.0] + [1.0] * len(spec.tools), 20)
                env._reset_idx(torch.tensor([0, 1]))
                self.assertAlmostEqual(env._xy[0, 0].item(), 3.0)
                self.assertAlmostEqual(env._yaw[1].item(), 1.2, places=5)
                self.assertTrue(torch.allclose(env._q, env._q_init.repeat(2, 1)))
                pose = env.robot.root_pose
                self.assertAlmostEqual(pose[1, 0].item(), 103.0, places=4)   # env origin offset
                self.assertAlmostEqual(pose[0, 2].item(), env._root_height, places=5)
                from heavy_machines import kinematics as K
                lo, hi = K.bounding_box(spec)
                reach = max(abs(v) for v in lo + hi) + 2.0
                for cam in env.cams.values():
                    self.assertTrue(torch.allclose(cam.orientations.norm(dim=1), torch.ones(2), atol=1e-5))
                    d = (cam.positions[:, :2] - pose[:, :2]).norm(dim=1)
                    self.assertTrue(bool((d < reach).all()), f"camera far from machine: {d.tolist()}")

    def test_random_spawn_in_region(self):
        env, spec = make_env(self.names[0])
        env.cfg.randomize_spawn = True
        for _ in range(20):
            env._reset_idx(torch.tensor([0, 1]))
            x0, y0, x1, y1 = env.cfg.spawn_region
            for e in range(2):
                self.assertTrue(x0 <= env._xy[e, 0].item() <= x1)
                self.assertTrue(y0 <= env._xy[e, 1].item() <= y1)

    def test_task_controller_completes_instructions(self):
        """collect_machine.py's scripted tasks must be achievable by every machine."""
        from heavy_machines.tasks import TaskController, instruction, sample_goals

        for name in self.names:
            for seed in range(4):
                with self.subTest(machine=name, seed=seed):
                    env, spec = make_env(name, num_envs=1, cameras=False)
                    goals = sample_goals(spec, random.Random(seed), 2, 4)
                    prompt = instruction(spec, goals)
                    ctrl = TaskController(spec, goals, env.cfg.sim.dt * env.cfg.decimation)
                    steps = 0
                    while not ctrl.done and steps < 30 * 120:
                        state = env._get_observations()["state"][0].tolist()
                        run(env, ctrl.act(state), 1)
                        steps += 1
                    self.assertTrue(ctrl.done, f"stuck on goal {ctrl.idx}: {prompt}")
                    self.assertEqual(ctrl.timeouts, 0, f"a goal timed out: {prompt}")
                    # every tool goal reached its target
                    for g in goals:
                        if g.kind == "tool":
                            j = spec.joint(spec.tools[g.tool].joint)
                            if j.type != "continuous":
                                self.assertTrue(j.lower - 1e-6 <= g.value <= j.upper + 1e-6)

    def test_observation_layout(self):
        for name in self.names:
            with self.subTest(machine=name):
                env, spec = make_env(name, cameras=False)
                obs = env._get_observations()
                self.assertEqual(obs["state"].shape, (2, len(spec.state_labels)))
                self.assertEqual(obs["base_pose_w"].shape, (2, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
