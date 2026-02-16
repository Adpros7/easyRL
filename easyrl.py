from __future__ import annotations

import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn


if TORCH_AVAILABLE:

    class _MLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


else:

    class _SimpleLinear:
        def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
            rnd = random.Random(seed)
            self.weight = [[rnd.uniform(-0.1, 0.1) for _ in range(out_dim)] for _ in range(in_dim)]
            self.bias = [0.0 for _ in range(out_dim)]

        def __call__(self, x: list[float]) -> list[float]:
            out: list[float] = []
            for j, b in enumerate(self.bias):
                total = b
                for i, xi in enumerate(x):
                    total += xi * self.weight[i][j]
                out.append(total)
            return out


@dataclass
class Transition:
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any


class EasyRL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float,
        gamma: float,
        use_target_network: bool = False,
        entropy_coef: float = 0.0,
        clip_grad_norm: float | None = None,
        device: str = "cpu",
        policy_net: Any | None = None,
        value_net: Any | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.use_target_network = use_target_network
        self.entropy_coef = entropy_coef
        self.clip_grad_norm = clip_grad_norm
        self.device = device
        self.transitions: list[Transition] = []
        self.steps = 0

        if TORCH_AVAILABLE:
            self._torch_device = torch.device(self.device)
            self.policy_net = (policy_net or _MLP(state_dim, action_dim)).to(self._torch_device)
            self.value_net = (value_net or _MLP(state_dim, 1)).to(self._torch_device)
            self.target_value_net = None
            if self.use_target_network:
                self.target_value_net = _MLP(state_dim, 1).to(self._torch_device)
                self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.optimizer = torch.optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=self.lr
            )
        else:
            self.policy_net = policy_net or _SimpleLinear(state_dim, action_dim, seed=0)
            self.value_net = value_net or _SimpleLinear(state_dim, 1, seed=1)
            self.target_value_net = None
            if self.use_target_network:
                self.target_value_net = _SimpleLinear(state_dim, 1, seed=2)
                self.target_value_net.weight = [row[:] for row in self.value_net.weight]
                self.target_value_net.bias = self.value_net.bias[:]
            self.optimizer = None

    def select_action(self, state: Any, deterministic: bool = False) -> int:
        if TORCH_AVAILABLE:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self._torch_device).reshape(1, -1)
            logits = self.policy_net(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            return int(action_t.item())

        s = [float(v) for v in state]
        logits = self.policy_net(s)
        if deterministic:
            return int(max(range(len(logits)), key=lambda idx: logits[idx]))
        probs = self._softmax(logits)
        r = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                return idx
        return len(probs) - 1

    def _softmax(self, logits: list[float]) -> list[float]:
        m = max(logits)
        exp_vals = [math.exp(x - m) for x in logits]
        s = sum(exp_vals)
        return [x / s for x in exp_vals]

    def step(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> dict[str, float]:
        if TORCH_AVAILABLE:
            return self._step_torch(state, action, reward, next_state, done)
        return self._step_simple(state, action, reward, next_state, done)

    def _step_torch(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> dict[str, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self._torch_device).reshape(1, -1)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self._torch_device).reshape(1, -1)
        action_t = torch.as_tensor(action, dtype=torch.long, device=self._torch_device).reshape(1)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self._torch_device).reshape(1)
        done_t = torch.as_tensor(float(done), dtype=torch.float32, device=self._torch_device).reshape(1)
        self.transitions.append(Transition(state_t, action_t, reward_t, next_state_t, done_t))

        value = self.value_net(state_t).squeeze(-1)
        with torch.no_grad():
            target_source = self.target_value_net if self.target_value_net is not None else self.value_net
            next_value = target_source(next_state_t).squeeze(-1)
            td_target = reward_t + (1.0 - done_t) * self.gamma * next_value

        td_error = td_target - value
        value_loss = td_error.pow(2).mean()

        logits = self.policy_net(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_t)
        entropy_bonus = dist.entropy().mean()
        policy_loss = -(log_prob * td_error.detach()).mean() - self.entropy_coef * entropy_bonus

        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()), self.clip_grad_norm
            )
        self.optimizer.step()

        if self.target_value_net is not None:
            self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.steps += 1
        return {"loss": float(loss.item()), "td_error": float(td_error.mean().item()), "steps": float(self.steps)}

    def _step_simple(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> dict[str, float]:
        s = [float(v) for v in state]
        ns = [float(v) for v in next_state]
        a = int(action)
        r = float(reward)
        d = float(done)
        self.transitions.append(Transition(s, a, r, ns, d))

        value = self.value_net(s)[0]
        target_net = self.target_value_net if self.target_value_net is not None else self.value_net
        next_value = target_net(ns)[0]
        td_target = r + (1.0 - d) * self.gamma * next_value
        td_error = td_target - value

        # value update
        for i, si in enumerate(s):
            self.value_net.weight[i][0] += self.lr * 2.0 * td_error * si
        self.value_net.bias[0] += self.lr * 2.0 * td_error

        # policy update
        logits = self.policy_net(s)
        probs = self._softmax(logits)
        for j in range(self.action_dim):
            grad_logit = (1.0 if j == a else 0.0) - probs[j]
            step_scale = self.lr * td_error * grad_logit
            for i, si in enumerate(s):
                self.policy_net.weight[i][j] += step_scale * si
            self.policy_net.bias[j] += step_scale

        if self.target_value_net is not None:
            self.target_value_net.weight = [row[:] for row in self.value_net.weight]
            self.target_value_net.bias = self.value_net.bias[:]

        self.steps += 1
        return {"loss": float(td_error * td_error), "td_error": float(td_error), "steps": float(self.steps)}

    def reset(self) -> None:
        self.transitions.clear()
        self.steps = 0

    def save(self, path: str | Path) -> None:
        if TORCH_AVAILABLE:
            payload = {
                "policy": self.policy_net.state_dict(),
                "value": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
            }
            torch.save(payload, path)
            return

        import json

        payload = {
            "policy_weight": self.policy_net.weight,
            "policy_bias": self.policy_net.bias,
            "value_weight": self.value_net.weight,
            "value_bias": self.value_net.bias,
            "steps": self.steps,
        }
        Path(path).write_text(json.dumps(payload))

    def load(self, path: str | Path) -> None:
        if TORCH_AVAILABLE:
            payload = torch.load(path, map_location=self._torch_device)
            self.policy_net.load_state_dict(payload["policy"])
            self.value_net.load_state_dict(payload["value"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.steps = int(payload.get("steps", 0))
            if self.target_value_net is not None:
                self.target_value_net.load_state_dict(self.value_net.state_dict())
            return

        import json

        payload = json.loads(Path(path).read_text())
        self.policy_net.weight = payload["policy_weight"]
        self.policy_net.bias = payload["policy_bias"]
        self.value_net.weight = payload["value_weight"]
        self.value_net.bias = payload["value_bias"]
        self.steps = int(payload["steps"])
        if self.target_value_net is not None:
            self.target_value_net.weight = [row[:] for row in self.value_net.weight]
            self.target_value_net.bias = self.value_net.bias[:]
