from typing import Tuple
import numpy as np

BBox = Tuple[float, float, float, float]


class SingleObjectTracker:
    """
    Simple tracker for a single object using a constant velocity Kalman filter.
    """

    def __init__(self, init_bbox: BBox) -> None:
        self.bbox: BBox = init_bbox

        # Initialize state: [cx, cy, vx, vy]
        cx = (init_bbox[0] + init_bbox[2]) / 2
        cy = (init_bbox[1] + init_bbox[3]) / 2
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=float)

        # State covariance
        self.P = np.eye(4)

        # Process noise
        self.Q = np.eye(4) * 0.01

        # Measurement noise
        self.R = np.eye(2) * 1.0

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # State transition (will update vx, vy)
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0

    def predict(self) -> Tuple[float, float]:
        # Predict next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx, cy = self.state[0], self.state[1]
        return cx, cy

    def update(self, new_bbox: BBox) -> None:
        # Measurement: object center
        cx = (new_bbox[0] + new_bbox[2]) / 2
        cy = (new_bbox[1] + new_bbox[3]) / 2
        z = np.array([cx, cy])

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Update bbox to match predicted center
        w = new_bbox[2] - new_bbox[0]
        h = new_bbox[3] - new_bbox[1]
        self.bbox = (
            self.state[0] - w / 2,
            self.state[1] - h / 2,
            self.state[0] + w / 2,
            self.state[1] + h / 2,
        )
