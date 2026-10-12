"""ROS 2 bag writing for heavy-machine episodes (no Isaac Sim, no rclpy).

Uses the pure-Python ``rosbags`` library.  ``EpisodeBagWriter`` turns the
environment's observation rows into the topics listed in collect_machine.py,
including the URDF and a full TF tree so RViz 2's RobotModel shows the
moving machine.
"""

from __future__ import annotations

import json

import numpy as np

from . import kinematics as K
from .spec import MachineSpec
from .urdf_builder import build_urdf, metadata

WORLD = "world"
LATCHED_TOPICS = ("/robot_description", "/tf_static", "/task", "/machine", "/spawn")


def typestore():
    from rosbags.typesys import Stores, get_typestore
    return get_typestore(Stores.ROS2_JAZZY)


def qos_overrides_yaml() -> str:
    """Make latched topics transient-local on `ros2 bag play` so late RViz subscribers get them."""
    return "".join(f"{t}: {{history: keep_last, depth: 1, reliability: reliable, durability: transient_local}}\n"
                   for t in LATCHED_TOPICS)


def joint_tf(jt, q: float):
    """Parent→child transform of one URDF joint at position q (what robot_state_publisher does)."""
    tf = K.Tf.from_xyz_rpy(jt.xyz, jt.rpy) @ K.joint_motion(jt, q)
    return tf.t, K.mat_to_quat_wxyz(tf.R)


class Msgs:
    """Message factories bound to a rosbags typestore."""

    def __init__(self, ts):
        self.ts, self.T = ts, ts.types

    def header(self, t_ns, frame):
        return self.T["std_msgs/msg/Header"](
            stamp=self.T["builtin_interfaces/msg/Time"](sec=int(t_ns // 1_000_000_000),
                                                       nanosec=int(t_ns % 1_000_000_000)),
            frame_id=frame)

    def string(self, s):
        return self.T["std_msgs/msg/String"](data=s)

    def vec3(self, v):
        return self.T["geometry_msgs/msg/Vector3"](x=float(v[0]), y=float(v[1]), z=float(v[2]))

    def quat(self, q):  # q = (w, x, y, z)
        return self.T["geometry_msgs/msg/Quaternion"](x=float(q[1]), y=float(q[2]), z=float(q[3]),
                                                     w=float(q[0]))

    def tf_stamped(self, t_ns, parent, child, pos, q):
        return self.T["geometry_msgs/msg/TransformStamped"](
            header=self.header(t_ns, parent), child_frame_id=child,
            transform=self.T["geometry_msgs/msg/Transform"](translation=self.vec3(pos), rotation=self.quat(q)))

    def tf(self, items):
        return self.T["tf2_msgs/msg/TFMessage"](transforms=items)

    def float_array(self, values, labels):
        D = self.T["std_msgs/msg/MultiArrayDimension"]
        dims = [D(label=",".join(labels), size=len(values), stride=len(values))]
        return self.T["std_msgs/msg/Float32MultiArray"](
            layout=self.T["std_msgs/msg/MultiArrayLayout"](dim=dims, data_offset=0),
            data=np.asarray(values, dtype=np.float32))

    def joint_state(self, t_ns, names, pos, vel):
        return self.T["sensor_msgs/msg/JointState"](
            header=self.header(t_ns, ""), name=list(names),
            position=np.asarray(pos, dtype=np.float64), velocity=np.asarray(vel, dtype=np.float64),
            effort=np.zeros(len(names), dtype=np.float64))

    def odom(self, t_ns, child, pos, q, lin, ang):
        P = self.T["geometry_msgs/msg/Pose"](position=self.T["geometry_msgs/msg/Point"](
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2])), orientation=self.quat(q))
        cov = np.zeros(36, dtype=np.float64)
        return self.T["nav_msgs/msg/Odometry"](
            header=self.header(t_ns, WORLD), child_frame_id=child,
            pose=self.T["geometry_msgs/msg/PoseWithCovariance"](pose=P, covariance=cov),
            twist=self.T["geometry_msgs/msg/TwistWithCovariance"](
                twist=self.T["geometry_msgs/msg/Twist"](linear=self.vec3(lin), angular=self.vec3(ang)),
                covariance=cov))

    def twist(self, lin, ang):
        return self.T["geometry_msgs/msg/Twist"](linear=self.vec3(lin), angular=self.vec3(ang))

    def image(self, t_ns, frame, rgb):
        h, w, _ = rgb.shape
        return self.T["sensor_msgs/msg/Image"](
            header=self.header(t_ns, frame), height=h, width=w, encoding="rgb8", is_bigendian=0,
            step=w * 3, data=np.ascontiguousarray(rgb).reshape(-1))


class EpisodeBagWriter:
    """Writes one episode into an open ``rosbags.rosbag2.Writer``.

    ``joint_names`` is the articulation joint order of ``joint_pos`` rows.
    """

    def __init__(self, writer, ts, spec: MachineSpec, joint_names: list[str]):
        self.w, self.ts, self.spec = writer, ts, spec
        self.msgs = Msgs(ts)
        self.names = list(joint_names)
        self.movable = {j.name: j for j in spec.movable_joints}
        self.conns: dict[str, object] = {}

    def _write(self, topic: str, msgtype: str, t_ns: int, msg):
        if topic not in self.conns:
            self.conns[topic] = self.w.add_connection(topic, msgtype, typestore=self.ts)
        self.w.write(self.conns[topic], t_ns, self.ts.serialize_cdr(msg, msgtype))

    def header(self, prompt: str, spawn: dict, t_ns: int = 0):
        """Latched episode-start messages: URDF, task, machine metadata, spawn, static TF."""
        m, spec = self.msgs, self.spec
        self._write("/robot_description", "std_msgs/msg/String", t_ns, m.string(build_urdf(spec)))
        self._write("/task", "std_msgs/msg/String", t_ns, m.string(prompt))
        self._write("/machine", "std_msgs/msg/String", t_ns, m.string(json.dumps(metadata(spec))))
        self._write("/spawn", "std_msgs/msg/String", t_ns, m.string(json.dumps(spawn)))
        static = [m.tf_stamped(t_ns, j.parent, j.child, *joint_tf(j, 0.0))
                  for j in spec.joints if j.type == "fixed"]
        for cam in spec.cameras:
            tf = K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)
            static.append(m.tf_stamped(t_ns, cam.link, f"cam_{cam.name}", tf.t, K.mat_to_quat_wxyz(tf.R)))
        self._write("/tf_static", "tf2_msgs/msg/TFMessage", t_ns, m.tf(static))

    def step(self, t_ns: int, state, action, joint_pos, joint_vel, base_pose_w, base_twist_body,
             images: dict | None = None):
        """One control step.  ``images`` maps camera name → HxWx3 uint8 array (or None to skip)."""
        m, spec = self.msgs, self.spec
        pose = list(base_pose_w)
        tfs = [m.tf_stamped(t_ns, WORLD, spec.root, pose[:3], pose[3:])]
        tfs += [m.tf_stamped(t_ns, self.movable[n].parent, self.movable[n].child,
                             *joint_tf(self.movable[n], joint_pos[i]))
                for i, n in enumerate(self.names) if n in self.movable]
        self._write("/tf", "tf2_msgs/msg/TFMessage", t_ns, m.tf(tfs))
        self._write("/joint_states", "sensor_msgs/msg/JointState", t_ns,
                    m.joint_state(t_ns, self.names, joint_pos, joint_vel))
        tw = list(base_twist_body)                      # REP 105: twist in the child (root) frame
        self._write(f"/{spec.name}/odom", "nav_msgs/msg/Odometry", t_ns,
                    m.odom(t_ns, spec.root, pose[:3], pose[3:], (tw[0], tw[1], 0.0), (0.0, 0.0, tw[2])))
        self._write(f"/{spec.name}/state", "std_msgs/msg/Float32MultiArray", t_ns,
                    m.float_array(state, spec.state_labels))
        self._write("/action", "std_msgs/msg/Float32MultiArray", t_ns,
                    m.float_array(action, spec.action_labels))
        self._write("/action/cmd", "geometry_msgs/msg/Twist", t_ns,
                    m.twist((action[0], 0, 0), (0, 0, action[1])))
        for cam, rgb in (images or {}).items():
            self._write(f"/camera/{cam}/image_raw", "sensor_msgs/msg/Image", t_ns,
                        m.image(t_ns, f"cam_{cam}", rgb))
