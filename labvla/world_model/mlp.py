from __future__ import annotations

import numpy as np


class LightweightWorldModel:
    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 64) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(0)
        self.w1 = rng.normal(0.0, 0.1, size=(state_dim + action_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w3 = rng.normal(0.0, 0.1, size=(hidden_dim, state_dim)).astype(np.float32)
        self.b3 = np.zeros((state_dim,), dtype=np.float32)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[-1] < self.action_dim:
            pad = np.zeros((action.shape[0], self.action_dim - action.shape[-1]), dtype=np.float32)
            action = np.concatenate([action, pad], axis=-1)
        elif action.shape[-1] > self.action_dim:
            action = action[:, : self.action_dim]
        x = np.concatenate([state, action], axis=-1)
        h1 = self._relu(x @ self.w1 + self.b1)
        h2 = self._relu(h1 @ self.w2 + self.b2)
        return h2 @ self.w3 + self.b3

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.forward(state, action)
