"""
Base Multi-Robot Environment - Pure NumPy Implementation (No PyBullet)

This provides a lightweight simulation environment that:
1. Works on any platform (Mac, Linux, Windows)
2. Has the same API as the PyBullet version
3. Simulates heterogeneous robot configurations
4. Supports different sensor modalities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class Robot:
    """Simple robot model with position, velocity, and sensor configuration."""
    
    def __init__(self, robot_id: int, position: np.ndarray, sensor_config: Dict):
        self.robot_id = robot_id
        self.position = position.astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.orientation = 0.0  # radians
        self.sensor_config = sensor_config
        
        # Sensor capabilities
        self.has_lidar = sensor_config.get('lidar', False)
        self.has_rgb = sensor_config.get('rgb', False)
        self.has_depth = sensor_config.get('depth', False)
        self.lidar_range = sensor_config.get('lidar_range', 5.0)
        self.lidar_rays = sensor_config.get('lidar_rays', 16)
        self.camera_fov = sensor_config.get('camera_fov', 60)
        
    def get_observation_dim(self) -> int:
        """Calculate observation dimension based on sensor configuration."""
        dim = 4  # Base: position (2) + velocity (2)
        if self.has_lidar:
            dim += self.lidar_rays
        if self.has_rgb:
            dim += 32  # Compressed RGB features
        if self.has_depth:
            dim += 16  # Compressed depth features
        return dim


class BaseMultiRobotEnv(ABC):
    """
    Base class for multi-robot environments using pure NumPy simulation.
    
    This provides a lightweight, portable simulation that works on any platform.
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        heterogeneity_level: int = 1,  # 0-3
        world_size: Tuple[float, float] = (20.0, 20.0),
        max_steps: int = 500,
        seed: Optional[int] = None
    ):
        self.num_robots = num_robots
        self.heterogeneity_level = heterogeneity_level
        self.world_size = world_size
        self.max_steps = max_steps
        self.step_count = 0

        # ------------------------------------------------------------------
        # Fixed-dimension observation interface
        # ------------------------------------------------------------------
        # A common reason for "all runs fail" (or pretrained checkpoints not
        # loading across heterogeneity levels) is a *changing* observation
        # dimension or a *changing* feature layout (e.g., omitting RGB features
        # when a robot lacks a camera, which shifts the indices of subsequent
        # features).
        #
        # To keep experiments comparable and checkpoints loadable, we enforce a
        # fixed observation layout across *all* robots and heterogeneity levels:
        #   [pos(2), vel(2), extra(k), lidar(L), rgb(32), depth(16)]
        # Missing modalities are filled with neutral defaults (see
        # _get_robot_observation).
        self._lidar_seg_dim = 16
        self._rgb_seg_dim = 32
        self._depth_seg_dim = 16
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
        self._seed = seed
        
        # Robot configurations based on heterogeneity level
        self.sensor_configs = self._generate_sensor_configs()
        
        # Initialize robots
        self.robots: List[Robot] = []
        self._init_robots()
        
        # Environment state
        self.obstacles: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self._init_environment()

        # Extra task-level observation features (environment-specific)
        # Subclasses can override _get_extra_obs_dim/_get_extra_obs.
        self.extra_obs_dim = int(self._get_extra_obs_dim())

        # Fixed observation dimension (constant across heterogeneity levels).
        # Base features: position(2) + velocity(2)
        base_dim = 4
        self.fixed_obs_dim = int(
            base_dim
            + self.extra_obs_dim
            + self._lidar_seg_dim
            + self._rgb_seg_dim
            + self._depth_seg_dim
        )

        # Maintain legacy names used elsewhere.
        self.max_obs_dim = self.fixed_obs_dim
        self.obs_dims = [self.fixed_obs_dim for _ in range(self.num_robots)]
        
    def _generate_sensor_configs(self) -> List[Dict]:
        """Generate sensor configurations based on heterogeneity level."""
        configs = []
        
        if self.heterogeneity_level == 0:
            # H0: Homogeneous - all robots have same sensors
            base_config = {
                'lidar': True, 'rgb': True, 'depth': True,
                'lidar_range': 5.0, 'lidar_rays': 16,
                'camera_fov': 60
            }
            configs = [base_config.copy() for _ in range(self.num_robots)]
            
        elif self.heterogeneity_level == 1:
            # H1: Mild heterogeneity - slight sensor variations
            for i in range(self.num_robots):
                config = {
                    'lidar': True,
                    'rgb': i % 2 == 0,  # Half have RGB
                    'depth': True,
                    'lidar_range': 4.0 + (i % 3),
                    'lidar_rays': 16,
                    'camera_fov': 50 + (i % 4) * 10
                }
                configs.append(config)
                
        elif self.heterogeneity_level == 2:
            # H2: Moderate heterogeneity - different sensor suites
            sensor_suites = [
                {'lidar': True, 'rgb': True, 'depth': False, 'lidar_range': 6.0, 'lidar_rays': 16},
                {'lidar': True, 'rgb': False, 'depth': True, 'lidar_range': 4.0, 'lidar_rays': 24},
                {'lidar': False, 'rgb': True, 'depth': True, 'lidar_range': 0, 'lidar_rays': 0},
                {'lidar': True, 'rgb': True, 'depth': True, 'lidar_range': 5.0, 'lidar_rays': 16},
            ]
            for i in range(self.num_robots):
                configs.append(sensor_suites[i % len(sensor_suites)].copy())
                
        else:  # H3
            # H3: Severe heterogeneity - very different capabilities
            sensor_suites = [
                {'lidar': True, 'rgb': False, 'depth': False, 'lidar_range': 8.0, 'lidar_rays': 32},
                {'lidar': False, 'rgb': True, 'depth': False, 'lidar_range': 0, 'lidar_rays': 0},
                {'lidar': False, 'rgb': False, 'depth': True, 'lidar_range': 0, 'lidar_rays': 0},
                {'lidar': True, 'rgb': True, 'depth': True, 'lidar_range': 3.0, 'lidar_rays': 8},
            ]
            for i in range(self.num_robots):
                configs.append(sensor_suites[i % len(sensor_suites)].copy())
                
        return configs
    
    def _init_robots(self):
        """Initialize robots with positions and configurations."""
        self.robots = []
        for i in range(self.num_robots):
            # Spawn robots in different quadrants
            quadrant = i % 4
            qx = (quadrant % 2) * self.world_size[0] / 2 + self.world_size[0] / 4
            qy = (quadrant // 2) * self.world_size[1] / 2 + self.world_size[1] / 4
            
            # Add some randomness within quadrant
            pos = np.array([
                qx + self.rng.uniform(-2, 2),
                qy + self.rng.uniform(-2, 2)
            ])
            
            robot = Robot(i, pos, self.sensor_configs[i])
            self.robots.append(robot)
    
    @abstractmethod
    def _init_environment(self):
        """Initialize environment-specific elements (obstacles, targets, etc.)."""
        pass

    # ---------------------------------------------------------------------
    # Optional task-level observation features
    # ---------------------------------------------------------------------
    def _get_extra_obs_dim(self) -> int:
        """Return the dimension of environment-specific, task-level features.

        Subclasses may override this to append extra features to each robot's
        observation (e.g., "carrying" flags, goal vectors, progress counters).
        """
        return 0

    def _get_extra_obs(self, robot: Robot) -> np.ndarray:
        """Return an array of length self.extra_obs_dim with extra features.

        Defaults to an empty vector.
        """
        return np.zeros((0,), dtype=np.float32)
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        
    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment and return initial observations."""
        self.step_count = 0
        self._init_robots()
        self._init_environment()
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], float, bool, Dict]:
        """
        Execute actions for all robots.
        
        Args:
            actions: List of action arrays, one per robot [vx, vy] or [v, omega]
            
        Returns:
            observations: List of observation arrays
            reward: Scalar team reward
            done: Boolean indicating episode termination
            info: Dictionary with additional information
        """
        self.step_count += 1
        
        # Apply actions to each robot
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            self._apply_action(robot, action)
        
        # Update environment state
        self._update_environment()
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate reward (scalar)
        reward = self._compute_reward()
        
        # Check termination
        done = self._check_done()
        
        # Get info
        info = self._get_info()
        
        return observations, reward, done, info
    
    def _apply_action(self, robot: Robot, action: np.ndarray):
            """Apply action to robot and update its state.

            Numerical safety
            ----------------
            In large sweeps (multiple methods × seeds × heterogeneity levels), a
            single NaN can cascade into crashes that invalidate the entire run.
            We therefore sanitize actions and keep robot state finite.

            - Non-finite actions are replaced with zeros.
            - If an update would produce a non-finite position, the robot does not move.
            """
            # Ensure action is a 2D vector [vx, vy]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape[0] != 2:
                if action.size >= 2:
                    action = action[:2]
                else:
                    action = np.zeros(2, dtype=np.float32)

            # Replace NaN/Inf (np.clip does not remove NaNs)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            # Ensure orientation is finite
            if not np.isfinite(robot.orientation):
                robot.orientation = 0.0

            # Simple velocity control
            max_speed = 1.0
            dt = 0.1
            velocity = action * max_speed
            if not np.all(np.isfinite(velocity)):
                velocity = np.zeros(2, dtype=np.float32)

            # Candidate position update
            new_pos = robot.position + velocity * dt
            if not np.all(np.isfinite(new_pos)):
                # Refuse the update rather than poisoning the state with NaNs
                robot.velocity = np.zeros(2, dtype=np.float32)
                return

            # Boundary collision
            new_pos = np.clip(new_pos, [0.0, 0.0], self.world_size).astype(np.float32)

            # Obstacle collision (simple check)
            collision = False
            for obs in self.obstacles:
                d = np.linalg.norm(new_pos - obs[:2])
                if not np.isfinite(d):
                    collision = True
                    break
                if d < obs[2]:  # obs = [x, y, radius]
                    collision = True
                    break

            if not collision:
                robot.position = new_pos

            robot.velocity = velocity.astype(np.float32)

            # Update orientation based on velocity
            vnorm = float(np.linalg.norm(robot.velocity))
            if np.isfinite(vnorm) and vnorm > 0.01:
                robot.orientation = float(np.arctan2(robot.velocity[1], robot.velocity[0]))

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all robots."""
        return [self._get_robot_observation(robot) for robot in self.robots]
    
    def _get_robot_observation(self, robot: Robot) -> np.ndarray:
        """Get a fixed-layout observation vector for a robot.

        Layout:
          [pos(2), vel(2), extra(k), lidar(L), rgb(32), depth(16)]

        Notes
        -----
        - All robots receive the *same* observation dimension. Robots without a
          modality get a neutral default for that segment.
        - Robots with different LiDAR ray counts are resampled to the fixed
          segment length L.
        """

        obs = np.zeros(self.fixed_obs_dim, dtype=np.float32)
        idx = 0

        # Base: position (normalized) + velocity
        pos_norm = robot.position / np.array(self.world_size, dtype=np.float32)
        obs[idx:idx + 2] = pos_norm.astype(np.float32)
        idx += 2
        obs[idx:idx + 2] = robot.velocity.astype(np.float32)
        idx += 2

        # Environment/task-level features (if any)
        if getattr(self, 'extra_obs_dim', 0) > 0:
            extra = self._get_extra_obs(robot)
            extra = np.asarray(extra, dtype=np.float32).reshape(-1)
            if extra.shape[0] != self.extra_obs_dim:
                raise ValueError(
                    f"Extra obs dim mismatch for robot {robot.robot_id}: "
                    f"expected {self.extra_obs_dim}, got {extra.shape[0]}"
                )
            obs[idx:idx + self.extra_obs_dim] = extra
        idx += int(self.extra_obs_dim)

        # LiDAR segment (distance-like) -> default to 1.0 (max range)
        lidar_seg = np.ones(self._lidar_seg_dim, dtype=np.float32)
        if bool(robot.has_lidar) and int(getattr(robot, 'lidar_rays', 0)) > 0:
            raw = self._simulate_lidar(robot)
            lidar_seg = self._resample_vector(raw, self._lidar_seg_dim)
        obs[idx:idx + self._lidar_seg_dim] = lidar_seg
        idx += self._lidar_seg_dim

        # RGB segment (feature-like) -> default zeros
        rgb_seg = np.zeros(self._rgb_seg_dim, dtype=np.float32)
        if bool(robot.has_rgb):
            rgb_seg = self._simulate_rgb(robot).astype(np.float32)
        obs[idx:idx + self._rgb_seg_dim] = rgb_seg
        idx += self._rgb_seg_dim

        # Depth segment (distance-like) -> default to 1.0 (max range)
        depth_seg = np.ones(self._depth_seg_dim, dtype=np.float32)
        if bool(robot.has_depth):
            depth_seg = self._simulate_depth(robot).astype(np.float32)
        obs[idx:idx + self._depth_seg_dim] = depth_seg

        return obs

    def _resample_vector(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        """Resample a 1D vector to a target length via linear interpolation."""
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        src_len = int(vec.shape[0])
        target_len = int(target_len)
        if src_len <= 0:
            return np.zeros(target_len, dtype=np.float32)
        if src_len == target_len:
            return vec.astype(np.float32)

        # Interpolate on normalized indices.
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float32)
        tgt_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
        return np.interp(tgt_x, src_x, vec).astype(np.float32)
    
    def _simulate_lidar(self, robot: Robot) -> np.ndarray:
        """Simulate LiDAR sensor readings."""
        # Numerical safety: if robot state is invalid, return max-range readings
        if not np.all(np.isfinite(robot.position)) or not np.isfinite(robot.orientation):
            return np.ones(int(getattr(robot, 'lidar_rays', 0)) or 1, dtype=np.float32)
        readings = np.ones(robot.lidar_rays, dtype=np.float32)  # Max range = 1.0 (normalized)
        
        angles = np.linspace(0, 2 * np.pi, robot.lidar_rays, endpoint=False)
        angles += robot.orientation
        
        for i, angle in enumerate(angles):
            # Cast ray
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            min_dist = robot.lidar_range
            
            # Check obstacles
            for obs in self.obstacles:
                dist = self._ray_circle_intersection(
                    robot.position, ray_dir, obs[:2], obs[2]
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Optionally detect task targets (e.g., packages/victims) as small circles
            if getattr(self, 'lidar_detects_targets', True) and getattr(self, 'targets', None):
                target_radius = float(getattr(self, 'lidar_target_radius', 0.4))
                for tgt in self.targets:
                    dist = self._ray_circle_intersection(
                        robot.position, ray_dir, np.asarray(tgt), target_radius
                    )
                    if dist is not None and dist < min_dist:
                        min_dist = dist
            
            # Check boundaries
            for boundary_dist in self._ray_boundary_intersection(robot.position, ray_dir):
                if boundary_dist < min_dist:
                    min_dist = boundary_dist
            
            denom = float(robot.lidar_range) if float(robot.lidar_range) > 1e-6 else 1e-6
            readings[i] = float(min_dist) / denom  # Normalize
        
        return readings
    
    def _ray_circle_intersection(self, origin: np.ndarray, direction: np.ndarray, 
                                  center: np.ndarray, radius: float) -> Optional[float]:
        """Calculate ray-circle intersection distance."""
        oc = origin - center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t > 0:
            return t
        return None
    
    def _ray_boundary_intersection(self, origin: np.ndarray, direction: np.ndarray) -> List[float]:
        """Calculate ray-boundary intersection distances."""
        distances = []
        
        # Check all four boundaries
        if direction[0] > 0:
            t = (self.world_size[0] - origin[0]) / direction[0]
            if t > 0:
                distances.append(t)
        elif direction[0] < 0:
            t = -origin[0] / direction[0]
            if t > 0:
                distances.append(t)
                
        if direction[1] > 0:
            t = (self.world_size[1] - origin[1]) / direction[1]
            if t > 0:
                distances.append(t)
        elif direction[1] < 0:
            t = -origin[1] / direction[1]
            if t > 0:
                distances.append(t)
        
        return distances
    
    def _simulate_rgb(self, robot: Robot) -> np.ndarray:
        """Simulate compressed RGB camera features."""
        # Numerical safety: invalid state -> no detections
        if not np.all(np.isfinite(robot.position)) or not np.isfinite(robot.orientation):
            return np.zeros(32, dtype=np.float32)
        # Simple feature extraction based on visible objects
        features = np.zeros(32, dtype=np.float32)
        
        # Encode nearby objects in feature space
        for i, target in enumerate(self.targets[:8]):  # Max 8 targets in features
            rel_pos = target - robot.position
            dist = np.linalg.norm(rel_pos)
            angle = np.arctan2(rel_pos[1], rel_pos[0]) - robot.orientation
            
            # Check if in camera FOV
            fov_rad = np.radians(robot.camera_fov)
            if abs(angle) < fov_rad / 2 and dist < 10.0:
                idx = i * 4
                features[idx] = dist / 10.0  # Normalized distance
                features[idx + 1] = np.cos(angle)
                features[idx + 2] = np.sin(angle)
                features[idx + 3] = 1.0  # Object detected
        
        return features
    
    def _simulate_depth(self, robot: Robot) -> np.ndarray:
        """Simulate compressed depth sensor features."""
        # Numerical safety: invalid state -> max depth
        if not np.all(np.isfinite(robot.position)) or not np.isfinite(robot.orientation):
            return np.ones(16, dtype=np.float32)
        # Simple depth map compressed to 16 values
        features = np.ones(16, dtype=np.float32)  # Max depth = 1.0
        
        angles = np.linspace(-np.radians(30), np.radians(30), 16)
        angles += robot.orientation
        
        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            min_dist = 10.0  # Max depth range
            
            for obs in self.obstacles:
                dist = self._ray_circle_intersection(robot.position, ray_dir, obs[:2], obs[2])
                if dist is not None and dist < min_dist:
                    min_dist = dist
            
            features[i] = min_dist / 10.0
        
        return features
    
    @abstractmethod
    def _update_environment(self):
        """Update environment state after actions."""
        pass
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute scalar team reward."""
        pass
    
    @abstractmethod
    def _check_done(self) -> bool:
        """Check if episode is done (boolean)."""
        pass
    
    @abstractmethod
    def _get_info(self) -> Dict:
        """Get additional information about environment state."""
        pass
    
    def get_observation_dim(self, robot_id: int = 0) -> int:
        """Get observation dimension for a specific robot."""
        return self.max_obs_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension (same for all robots)."""
        return 2  # [vx, vy]
    
    def close(self):
        """Clean up environment resources."""
        pass  # No resources to clean up in NumPy version
