from collections import deque
import os

import gymnasium as gym
import numpy as np


class ProjectionMapWrapper(gym.Wrapper):
    def __init__(self, env, enabled=False, history_len=120, window_name="quad projection map",
                 show_obstacle_point_cloud=False):
        gym.Wrapper.__init__(self, env)
        self.enabled = enabled
        self.history_len = history_len
        self.window_name = window_name
        self.show_obstacle_point_cloud = show_obstacle_point_cloud
        self.path_history = None
        self._display_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        base_env = self.env.unwrapped
        num_agents = getattr(base_env, "num_agents", 1)
        self.path_history = [deque(maxlen=self.history_len) for _ in range(num_agents)]
        self._append_positions()
        return obs, info

    def step(self, action):
        obs, reward, info, terminated, truncated = self.env.step(action)
        self._append_positions()
        return obs, reward, info, terminated, truncated

    def render(self):
        frame = self.env.render()
        if not self.enabled:
            return frame

        map_frame = self._render_projection_map(base_frame=frame)
        if map_frame is None:
            return frame

        if frame is not None:
            try:
                import cv2

                if map_frame.shape[0] != frame.shape[0]:
                    map_frame = cv2.resize(map_frame, (map_frame.shape[1], frame.shape[0]))
                return np.concatenate((frame, map_frame), axis=1)
            except Exception:
                return frame

        if self._display_enabled:
            try:
                import cv2

                cv2.imshow(self.window_name, cv2.cvtColor(map_frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            except Exception:
                pass

        return frame

    def _append_positions(self):
        if self.path_history is None:
            return

        base_env = self.env.unwrapped
        positions = getattr(base_env, "pos", None)
        if positions is None:
            positions = np.array([e.dynamics.pos for e in base_env.envs])
        for i, pos in enumerate(np.asarray(positions)):
            if i < len(self.path_history):
                self.path_history[i].append(np.array(pos[:3], dtype=float))

    def _render_projection_map(self, base_frame=None):
        try:
            import cv2
        except Exception:
            return None

        base_env = self.env.unwrapped
        room_dims = np.asarray(getattr(base_env, "room_dims", np.array([10.0, 10.0, 10.0])), dtype=float)
        width = base_frame.shape[1] // 2 if base_frame is not None else 640
        height = base_frame.shape[0] if base_frame is not None else 640
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        x_min, y_min = -room_dims[0] / 2.0, -room_dims[1] / 2.0
        x_max, y_max = room_dims[0] / 2.0, room_dims[1] / 2.0

        def to_px(pt):
            x, y = float(pt[0]), float(pt[1])
            px = int((x - x_min) / max(1e-6, x_max - x_min) * (width - 1))
            py = int((y_max - y) / max(1e-6, y_max - y_min) * (height - 1))
            return np.array([px, py], dtype=int)

        def draw_polyline(points, color, thickness=2):
            if len(points) < 2:
                return
            pts = np.array([to_px(p) for p in points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 2)

        obstacles = getattr(getattr(base_env, "obstacles", None), "pos_arr", None)
        obstacle_size = float(getattr(base_env, "obst_size", 0.6))
        if obstacles is not None and len(obstacles) > 0:
            radius_px = max(2, int(obstacle_size / max(room_dims[0], room_dims[1]) * width * 0.5))
            for obst in np.asarray(obstacles):
                center = tuple(to_px(obst))
                cv2.circle(canvas, center, radius_px, (70, 150, 70), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, center, radius_px, (40, 90, 40), 1, lineType=cv2.LINE_AA)

        positions = getattr(base_env, "pos", None)
        if positions is None:
            positions = np.array([e.dynamics.pos for e in base_env.envs])

        if self.show_obstacle_point_cloud:
            self._draw_obstacle_point_cloud(canvas, to_px, np.asarray(positions), room_dims)

        spawn_points = getattr(getattr(base_env, "scenario", None), "spawn_points", None)
        if spawn_points is not None:
            for spawn in np.asarray(spawn_points):
                center = tuple(to_px(spawn))
                cv2.drawMarker(canvas, center, (140, 140, 140), markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=8, thickness=1, line_type=cv2.LINE_AA)

        goals = [e.goal for e in base_env.envs]
        for idx, goal in enumerate(goals):
            center = tuple(to_px(goal))
            color = (40 + (idx * 67) % 160, 30, 220)
            cv2.drawMarker(canvas, center, color, markerType=cv2.MARKER_STAR, markerSize=16,
                           thickness=2, line_type=cv2.LINE_AA)

        for idx, pos in enumerate(np.asarray(positions)):
            trail = list(self.path_history[idx]) if self.path_history is not None else []
            draw_polyline(trail, (120, 120, 255), thickness=2)
            center = tuple(to_px(pos))
            color = (255, 90, 50) if idx == 0 else (255, 160, 60)
            cv2.circle(canvas, center, 6, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center, 8, (20, 20, 20), 1, lineType=cv2.LINE_AA)

        cv2.putText(canvas, "XY projection", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2,
                    cv2.LINE_AA)
        return canvas

    def _draw_obstacle_point_cloud(self, canvas, to_px, positions, room_dims):
        try:
            import cv2
        except Exception:
            return

        base_env = self.env.unwrapped
        obstacles = getattr(getattr(base_env, "obstacles", None), "pos_arr", None)
        obstacle_radius = float(getattr(base_env, "obst_size", 0.6)) / 2.0
        scan_resolution = float(getattr(base_env, "obstacle_scan_resolution", 0.25))
        scan_radius = max(2.0, scan_resolution * 8.0)
        cloud_color = (30, 210, 210)
        wall_color = (210, 120, 30)

        def near_any_quad(point):
            if positions.size == 0:
                return False
            deltas = positions[:, :2] - np.asarray(point[:2], dtype=float)
            return np.any(np.linalg.norm(deltas, axis=1) <= scan_radius)

        if obstacles is not None and len(obstacles) > 0:
            angles = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
            for obst in np.asarray(obstacles):
                for angle in angles:
                    point = np.array([
                        obst[0] + obstacle_radius * np.cos(angle),
                        obst[1] + obstacle_radius * np.sin(angle),
                    ])
                    if near_any_quad(point):
                        cv2.circle(canvas, tuple(to_px(point)), 2, cloud_color, -1, lineType=cv2.LINE_AA)

        half_length = room_dims[0] / 2.0
        half_width = room_dims[1] / 2.0
        step = max(0.1, scan_resolution)
        xs = np.arange(-half_length, half_length + step, step)
        ys = np.arange(-half_width, half_width + step, step)
        for x in xs:
            for point in ((x, -half_width), (x, half_width)):
                if near_any_quad(point):
                    cv2.circle(canvas, tuple(to_px(point)), 2, wall_color, -1, lineType=cv2.LINE_AA)
        for y in ys:
            for point in ((-half_length, y), (half_length, y)):
                if near_any_quad(point):
                    cv2.circle(canvas, tuple(to_px(point)), 2, wall_color, -1, lineType=cv2.LINE_AA)

        for pos in positions:
            center = tuple(to_px(pos))
            radius_px = max(8, int(scan_radius / max(room_dims[0], room_dims[1]) * canvas.shape[1]))
            cv2.circle(canvas, center, radius_px, (180, 180, 180), 1, lineType=cv2.LINE_AA)
