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
                goals.append(Goal("tool", tool_q[i] + math.radians(deg), i,
                                  f"{t.label.replace('_', ' ')} "
                                  f"{(t.positive if deg > 0 else t.negative) or ('+' if deg > 0 else '-')} "
                                  f"{abs(deg)} degrees"))
                tool_q[i] += math.radians(deg)
            else:
                cur = _pct(spec, i, tool_q[i])
                choices = [p for p in range(0, 101, 10) if abs(p - cur) >= 30]
                pct = rng.choice(choices or [0, 100])
                q = j.lower + pct / 100 * (j.upper - j.lower)
                word = t.positive if q > tool_q[i] else t.negative
                verb = f"{word} the {t.label.replace('_', ' ')}" if word else f"move the {t.label}"
                goals.append(Goal("tool", q, i, f"{verb} to {pct}%"))
                tool_q[i] = q
    return goals


def instruction(spec: MachineSpec, goals: list[Goal]) -> str:
    return f"{spec.display_name.lower()}: " + ", then ".join(g.text for g in goals)


class TaskController:
    """Closed-loop controller that executes goals one after another.

    ``act(state)`` takes the env state row ``[x, y, yaw, speed, yaw_rate,
    steer, tool_1..tool_k]`` and returns an action list in [-1, 1].
    """

    def __init__(self, spec: MachineSpec, goals: list[Goal], control_dt: float):
        self.spec, self.goals, self.dt = spec, goals, control_dt
        self.idx = 0
        self._start: list[float] | None = None
        self._unwrapped: dict[int, float] = {}
        self._prev_tool: dict[int, float] = {}
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
        self.steps_in_goal = 0

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

    def act(self, state: list[float]) -> list[float]:
        spec, d = self.spec, self.spec.drive
        a = [0.0] * spec.action_dim
        tools = [self._tool_pos(i, state[6 + i]) for i in range(len(spec.tools))]
        while not self.done:
            g = self.goals[self.idx]
            if self._start is None:
                self._start = list(state[:3]) + tools
            self.steps_in_goal += 1
            if self.steps_in_goal > 30 * 40:              # 40 s safety timeout per goal
                self.timeouts += 1
                self._advance()
                continue
            if g.kind == "drive":
                dist = math.hypot(state[0] - self._start[0], state[1] - self._start[1])
                remaining = abs(g.value) - dist
                if remaining <= 0.05:
                    self._advance()
                    continue
                a[0] = math.copysign(min(1.0, 0.25 + remaining / 3.0), g.value)
                return a
            if g.kind == "turn":
                turned = math.atan2(math.sin(state[2] - self._start[2]), math.cos(state[2] - self._start[2]))
                remaining = abs(g.value) - abs(turned) if turned * g.value >= 0 else abs(g.value)
                if remaining <= math.radians(2):
                    self._advance()
                    continue
                a[1] = math.copysign(1.0, g.value)
                if d.type != "skid":                     # wheeled machines must roll to turn
                    a[0] = 0.5
                return a
            # tool goal
            err = g.value - tools[g.tool]
            j = spec.joint(spec.tools[g.tool].joint)
            tol = 0.01 if j.type == "prismatic" else math.radians(1.0)
            if abs(err) <= tol:
                self._advance()
                continue
            a[2 + g.tool] = max(-1.0, min(1.0, err / (j.velocity * self.dt * 6)))
            return a
        return a
