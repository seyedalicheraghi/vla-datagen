"""Language-conditioned scripted tasks for any machine (pure Python).

An episode is a short random sequence of goal primitives, e.g.

    "hydraulic excavator: swing left 90 degrees, then raise the boom to 70%,
     then drive forward 4.0 m"

The instruction is fixed at the start of the episode and a closed-loop
controller (:class:`TaskController`) produces actions until every goal is
reached, so the language, the actions and the outcome always agree.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .spec import MachineSpec


@dataclass
class Goal:
    kind: str                 # "drive" | "turn" | "tool"
    value: float              # metres (drive), radians (turn), target position (tool)
    tool: int = -1            # tool index for kind == "tool"
    text: str = ""


# Scripted motion stays at realistic speeds even for road-going machines.
DRIVE_SPEED_CAP = 4.0      # m/s
TURN_SPEED_CAP = 2.5       # m/s while turning (wheeled machines)


def tool_phrase(label: str, word: str) -> str:
    """'raise the boom', 'extend the boom telescope', 'swing the chute left', 'curl the bucket'."""
    name = label.replace("_", " ")
    w = (word or "").lower()
    if w in ("raise", "up", "lift"):
        return f"raise the {name}"
    if w in ("lower", "down"):
        return f"lower the {name}"
    if w in ("extend", "out"):
        return f"extend the {name}"
    if w in ("retract", "in"):
        return f"retract the {name}"
    if w in ("left", "ccw"):
        return f"swing the {name} left" if w == "left" else f"turn the {name} counter-clockwise"
    if w in ("right", "cw"):
        return f"swing the {name} right" if w == "right" else f"turn the {name} clockwise"
    if not w:
        return f"move the {name}"
    return f"{w.replace('_', ' ')} the {name}"


def _pct(spec: MachineSpec, i: int, q: float) -> int:
    j = spec.joint(spec.tools[i].joint)
    return round(100 * (q - j.lower) / (j.upper - j.lower))


def sample_goals(spec: MachineSpec, rng: random.Random, n_min: int = 2, n_max: int = 4,
                 init_tools: list[float] | None = None) -> list[Goal]:
    """Random goal sequence; tool targets are sampled away from the current value."""
    tool_q = list(init_tools) if init_tools else [spec.joint(t.joint).init for t in spec.tools]
    goals: list[Goal] = []
    for _ in range(rng.randint(n_min, n_max)):
        kinds = ["drive", "turn"] + ["tool"] * (2 * len(spec.tools))
        kind = rng.choice(kinds)
        if kind == "drive":
            d = rng.choice([-1, 1, 1]) * round(rng.uniform(1.5, 6.0), 1)
            goals.append(Goal("drive", d, text=f"drive {'forward' if d > 0 else 'backward'} {abs(d):.1f} m"))
        elif kind == "turn":
            deg = rng.choice([-1, 1]) * rng.choice([30, 45, 60, 90])
            goals.append(Goal("turn", math.radians(deg),
                              text=f"turn {'left' if deg > 0 else 'right'} {abs(deg)} degrees"))
        else:
            i = rng.randrange(len(spec.tools))
            t, j = spec.tools[i], spec.joint(spec.tools[i].joint)
            if j.type == "continuous":
                deg = rng.choice([-1, 1]) * rng.choice([45, 90, 135, 180])
                word = t.positive if deg > 0 else t.negative
                goals.append(Goal("tool", tool_q[i] + math.radians(deg), i,
                                  f"{tool_phrase(t.label, word)} {abs(deg)} degrees"))
                tool_q[i] += math.radians(deg)
            else:
                cur = _pct(spec, i, tool_q[i])
                choices = [p for p in range(0, 101, 10) if abs(p - cur) >= 30]
                pct = rng.choice(choices or [0, 100])
                q = j.lower + pct / 100 * (j.upper - j.lower)
                word = t.positive if q > tool_q[i] else t.negative
                goals.append(Goal("tool", q, i, f"{tool_phrase(t.label, word)} to {pct}%"))
                tool_q[i] = q
    return goals


def instruction(spec: MachineSpec, goals: list[Goal]) -> str:
    return f"{spec.display_name.lower()}: " + ", then ".join(g.text for g in goals)


class TaskController:
    """Closed-loop controller that executes goals one after another.

    ``act(state)`` takes the env state row ``[x, y, yaw, speed, yaw_rate,
    steer, tool_1..tool_k]`` and returns an action list in [-1, 1].
    Wheeled machines re-centre their steering (standing still) after every
    turn, so a following "drive forward" really goes straight.
    """

    def __init__(self, spec: MachineSpec, goals: list[Goal], control_dt: float):
        self.spec, self.goals, self.dt = spec, goals, control_dt
        self.idx = 0
        self._start: list[float] | None = None
        self._unwrapped: dict[int, float] = {}
        self._prev_tool: dict[int, float] = {}
        self._prev_yaw: float | None = None
        self._turned = 0.0
        self._settling = False
        self.steps_in_goal = 0
        self.timeouts = 0                 # goals abandoned by the safety timeout

    @property
    def done(self) -> bool:
        return self.idx >= len(self.goals)

    @property
    def current(self) -> Goal | None:
        return None if self.done else self.goals[self.idx]

    def _advance(self):
        self.idx += 1
        self._start = None
        self._turned = 0.0
        self._settling = False
        self.steps_in_goal = 0

    def _timeout_steps(self, g: Goal) -> int:
        secs = 40.0
        if g.kind == "tool":
            j = self.spec.joint(self.spec.tools[g.tool].joint)
            span = 2 * math.pi if j.type == "continuous" else (j.upper - j.lower)
            secs = 20.0 + 2.0 * span / j.velocity
        return int(secs / self.dt)

    def _tool_pos(self, i: int, raw: float) -> float:
        """Unwrap continuous tools so targets beyond ±pi are reachable."""
        if self.spec.joint(self.spec.tools[i].joint).type != "continuous":
            return raw
        prev = self._prev_tool.get(i)
        un = self._unwrapped.get(i, raw)
        if prev is not None:
            d = math.atan2(math.sin(raw - prev), math.cos(raw - prev))
            un += d
        self._prev_tool[i], self._unwrapped[i] = raw, un
        return un

    def _speed_cmd(self, v: float) -> float:
        d = self.spec.drive
        return max(0.0, min(1.0, v / d.max_speed))

    def act(self, state: list[float]) -> list[float]:
        spec, d = self.spec, self.spec.drive
        a = [0.0] * spec.action_dim
        tools = [self._tool_pos(i, state[6 + i]) for i in range(len(spec.tools))]
        if self._prev_yaw is not None:
            self._turned += math.atan2(math.sin(state[2] - self._prev_yaw), math.cos(state[2] - self._prev_yaw))
        self._prev_yaw = state[2]
        while not self.done:
            g = self.goals[self.idx]
            if self._start is None:
                self._start = list(state[:3]) + tools
                self._turned = 0.0
            self.steps_in_goal += 1
            if self.steps_in_goal > self._timeout_steps(g):
                self.timeouts += 1
                self._advance()
                continue
            if g.kind == "drive":
                dist = math.hypot(state[0] - self._start[0], state[1] - self._start[1])
                remaining = abs(g.value) - dist
                if remaining <= 0.05:
                    self._advance()
                    continue
                v = min(DRIVE_SPEED_CAP, d.max_speed if g.value > 0 else d.reverse_speed,
                        0.4 + 1.2 * remaining)
                a[0] = math.copysign(max(0.05, self._speed_cmd(v)), g.value)
                return a
            if g.kind == "turn":
                if self._settling:                       # re-centre steering standing still
                    if abs(state[5]) <= 0.02:
                        self._advance()
                        continue
                    return a
                remaining = abs(g.value) - math.copysign(1.0, g.value) * self._turned
                if d.type == "articulated":
                    # Straightening the hinge at standstill swings the front frame back
                    # by about rear_length/(front_length+rear_length) of the hinge angle.
                    remaining -= d.rear_length * abs(state[5]) / (d.front_length + d.rear_length)
                if remaining <= math.radians(1.5):
                    if d.type == "skid":
                        self._advance()
                        continue
                    self._settling = True
                    return a
                a[1] = math.copysign(1.0, g.value)
                if d.type != "skid":                     # wheeled machines must roll to turn
                    a[0] = self._speed_cmd(min(TURN_SPEED_CAP, d.max_speed))
                return a
            # tool goal
            err = g.value - tools[g.tool]
            j = spec.joint(spec.tools[g.tool].joint)
            if j.type == "continuous":
                tol = math.radians(1.0)
            else:
                span = j.upper - j.lower
                tol = min(0.01 if j.type == "prismatic" else math.radians(1.0), 0.02 * span)
            if abs(err) <= tol:
                self._advance()
                continue
            a[2 + g.tool] = max(-1.0, min(1.0, err / (j.velocity * self.dt * 6)))
            return a
        return a
