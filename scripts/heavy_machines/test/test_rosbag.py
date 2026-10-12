"""Bag round trip: write an episode with rosbag_io, read it back, check topics and TF.

Needs torch and rosbags (skipped otherwise); no Isaac Sim.

    python scripts/heavy_machines/test/test_rosbag.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import torch  # noqa: F401
    from rosbags.rosbag2 import Reader, StoragePlugin, Writer
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False


def _quat_to_mat(w, x, y, z):
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


@unittest.skipUnless(HAVE_DEPS, "needs torch, numpy and rosbags")
class TestBagRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_env_logic as harness

        harness._install_isaac_stubs()
        cls.h = harness
        from heavy_machines import registry
        cls.names = registry.names()

    def test_every_machine(self):
        from heavy_machines import kinematics as K
        from heavy_machines.rosbag_io import LATCHED_TOPICS, WORLD, EpisodeBagWriter, typestore
        from heavy_machines.tasks import TaskController, instruction, sample_goals
        from heavy_machines.urdf_builder import build_urdf

        ts = typestore()
        for name in self.names:
            with self.subTest(machine=name), tempfile.TemporaryDirectory() as tmp:
                env, spec = self.h.make_env(name, num_envs=1, cameras=False)
                env._reset_idx(torch.tensor([0]))
                goals = sample_goals(spec, random.Random(1), 2, 3)
                prompt = instruction(spec, goals)
                ctrl = TaskController(spec, goals, env.cfg.sim.dt * env.cfg.decimation)
                steps, path = 40, os.path.join(tmp, "ep")
                with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as w:
                    bag = EpisodeBagWriter(w, ts, spec, env.robot.joint_names)
                    bag.header(prompt, {"episode": 0, **env.last_spawn[0]})
                    for k in range(steps):
                        obs = env._get_observations()
                        state = obs["state"][0].tolist()
                        action = ctrl.act(state)
                        imgs = ({c.name: np.zeros((8, 8, 3), np.uint8) for c in spec.cameras}
                                if k % 10 == 0 else None)
                        bag.step(k * 33_333_333, state, action, obs["joint_pos"][0].tolist(),
                                 obs["joint_vel"][0].tolist(), obs["base_pose_w"][0].tolist(),
                                 obs["base_twist_body"][0].tolist(), imgs)
                        self.h.run(env, action, 1)
                    last_q = obs["joint_pos"][0].tolist()
                    last_pose = obs["base_pose_w"][0].tolist()

                msgs: dict[str, list] = {}
                with Reader(path) as r:
                    for conn, _, raw in r.messages():
                        msgs.setdefault(conn.topic, []).append(ts.deserialize_cdr(raw, conn.msgtype))

                for t in LATCHED_TOPICS:
                    self.assertEqual(len(msgs.get(t, [])), 1, t)
                self.assertEqual(msgs["/robot_description"][0].data, build_urdf(spec))
                self.assertEqual(msgs["/task"][0].data, prompt)
                for t in ("/tf", "/joint_states", f"/{name}/odom", f"/{name}/state", "/action", "/action/cmd"):
                    self.assertEqual(len(msgs[t]), steps, t)
                for c in spec.cameras:
                    self.assertEqual(len(msgs[f"/camera/{c.name}/image_raw"]), 4)
                self.assertEqual(list(msgs["/joint_states"][-1].name), list(env.robot.joint_names))

                # Every URDF link must be reachable from `world` through /tf + /tf_static.
                edges = {}
                for m in msgs["/tf_static"][0].transforms + msgs["/tf"][-1].transforms:
                    tr = m.transform
                    edges[m.child_frame_id] = (m.header.frame_id, K.Tf(
                        _quat_to_mat(tr.rotation.w, tr.rotation.x, tr.rotation.y, tr.rotation.z),
                        (tr.translation.x, tr.translation.y, tr.translation.z)))

                def to_world(frame):
                    tf = K.Tf()
                    while frame != WORLD:
                        self.assertIn(frame, edges, f"frame {frame} has no TF parent")
                        parent, t = edges[frame]
                        tf, frame = t @ tf, parent
                    return tf

                for lk in spec.links:
                    to_world(lk.name)

                # Camera frames from TF must match base pose ∘ FK ∘ mount (what the env renders from).
                qdict = dict(zip(env.robot.joint_names, last_q))
                fk = K.link_transforms(spec, qdict)
                base = K.Tf(_quat_to_mat(*last_pose[3:]), tuple(last_pose[:3]))
                for c in spec.cameras:
                    want = base @ fk[c.link] @ K.Tf.from_xyz_rpy(c.xyz, c.rpy)
                    got = to_world(f"cam_{c.name}")
                    for a, b in zip(got.t, want.t):
                        self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
