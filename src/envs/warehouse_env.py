"""
Multi-Robot Warehouse Environment with Heterogeneous Sensors

This environment simulates a warehouse navigation task where multiple robots
with different sensor configurations must collaborate to complete deliveries.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import math


class WarehouseEnv(gym.Env):
    """
    A multi-robot warehouse environment for testing data-centric adaptation.
    
    Features:
    - N robots with heterogeneous sensor configurations
    - Configurable sensor noise and degradation
    - Package pickup and delivery tasks
    - Collision detection and avoidance
    - Team-level reward signal
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        num_robots: int = 4,
        arena_size: float = 20.0,
        max_steps: int = 500,
        num_packages: int = 8,
        num_obstacles: int = 10,
        sensor_configs: Optional[List[Dict]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_robots = num_robots
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        # Set random seed
        self.np_random = np.random.default_rng(seed)
        
        # Default sensor configurations for heterogeneous robots
        if sensor_configs is None:
            self.sensor_configs = self._default_sensor_configs()
        else:
            self.sensor_configs = sensor_configs
        
        # Observation and action spaces
        # Each robot observes: [lidar (32), camera_features (16), odometry (4), package_info (4)]
        self.obs_dim = 56
        self.action_dim = 2  # [linear_velocity, angular_velocity]
        
        # Multi-agent observation space
        self.observation_space = spaces.Dict({
            f"robot_{i}": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            ) for i in range(num_robots)
        })
        
        # Multi-agent action space
        self.action_space = spaces.Dict({
            f"robot_{i}": spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            ) for i in range(num_robots)
        })
        
        # Environment state
        self.robot_positions = None
        self.robot_orientations = None
        self.robot_velocities = None
        self.packages = None
        self.delivery_zones = None
        self.obstacles = None
        self.carried_packages = None
        self.delivered_count = 0
        self.collision_count = 0
        self.step_count = 0
        
    def _default_sensor_configs(self) -> List[Dict]:
        """Create default heterogeneous sensor configurations."""
        configs = [
            {  # Robot 0: Full sensors, healthy
                "lidar_enabled": True,
                "lidar_noise": 0.02,
                "lidar_range": 5.5,
                "camera_enabled": True,
                "camera_noise": 0.05,
                "odometry_noise": 0.01
            },
            {  # Robot 1: Camera-only
                "lidar_enabled": False,
                "lidar_noise": 0.0,
                "lidar_range": 0.0,
                "camera_enabled": True,
                "camera_noise": 0.05,
                "odometry_noise": 0.01
            },
            {  # Robot 2: LiDAR-only, degraded
                "lidar_enabled": True,
                "lidar_noise": 0.15,  # High noise (degraded sensor)
                "lidar_range": 3.0,   # Reduced range
                "camera_enabled": False,
                "camera_noise": 0.0,
                "odometry_noise": 0.02
            },
            {  # Robot 3: Full sensors, noisy
                "lidar_enabled": True,
                "lidar_noise": 0.08,
                "lidar_range": 5.0,
                "camera_enabled": True,
                "camera_noise": 0.1,
                "odometry_noise": 0.03
            }
        ]
        return configs[:self.num_robots]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Initialize robot positions (spread around the arena)
        self.robot_positions = np.zeros((self.num_robots, 2))
        self.robot_orientations = np.zeros(self.num_robots)
        self.robot_velocities = np.zeros((self.num_robots, 2))
        
        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            radius = self.arena_size * 0.3
            self.robot_positions[i] = [
                self.arena_size / 2 + radius * np.cos(angle),
                self.arena_size / 2 + radius * np.sin(angle)
            ]
            self.robot_orientations[i] = angle + np.pi  # Face center
        
        # Initialize obstacles (static)
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = self.np_random.uniform(2, self.arena_size - 2, size=2)
            size = self.np_random.uniform(0.5, 1.5, size=2)
            self.obstacles.append({"pos": pos, "size": size})
        
        # Initialize packages
        self.packages = []
        for _ in range(self.num_packages):
            pos = self._sample_free_position()
            self.packages.append({"pos": pos, "picked": False, "delivered": False})
        
        # Initialize delivery zones
        self.delivery_zones = [
            {"pos": np.array([2.0, 2.0]), "radius": 1.5},
            {"pos": np.array([self.arena_size - 2, 2.0]), "radius": 1.5},
            {"pos": np.array([2.0, self.arena_size - 2]), "radius": 1.5},
            {"pos": np.array([self.arena_size - 2, self.arena_size - 2]), "radius": 1.5}
        ]
        
        # Reset counters
        self.carried_packages = [None] * self.num_robots
        self.delivered_count = 0
        self.collision_count = 0
        self.step_count = 0
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _sample_free_position(self) -> np.ndarray:
        """Sample a position that doesn't collide with obstacles."""
        for _ in range(100):
            pos = self.np_random.uniform(3, self.arena_size - 3, size=2)
            if not self._check_obstacle_collision(pos, 0.5):
                return pos
        return self.np_random.uniform(3, self.arena_size - 3, size=2)
    
    def _check_obstacle_collision(self, pos: np.ndarray, radius: float) -> bool:
        """Check if a position collides with any obstacle."""
        for obs in self.obstacles:
            obs_center = obs["pos"]
            obs_half = obs["size"] / 2
            # Simple AABB collision
            if (pos[0] - radius < obs_center[0] + obs_half[0] and
                pos[0] + radius > obs_center[0] - obs_half[0] and
                pos[1] - radius < obs_center[1] + obs_half[1] and
                pos[1] + radius > obs_center[1] - obs_half[1]):
                return True
        return False
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Process actions for each robot
        for i in range(self.num_robots):
            action = actions[f"robot_{i}"]
            self._move_robot(i, action)
        
        # Check for package pickups and deliveries
        self._process_packages()
        
        # Calculate team reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self.delivered_count >= self.num_packages
        truncated = self.step_count >= self.max_steps
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, reward, terminated, truncated, info
    
    def _move_robot(self, robot_idx: int, action: np.ndarray):
        """Move a robot based on action."""
        linear_vel = np.clip(action[0], -1, 1) * 0.5  # Max 0.5 m/s
        angular_vel = np.clip(action[1], -1, 1) * 1.0  # Max 1.0 rad/s
        
        dt = 0.1  # Time step
        
        # Update orientation
        self.robot_orientations[robot_idx] += angular_vel * dt
        self.robot_orientations[robot_idx] = self.robot_orientations[robot_idx] % (2 * np.pi)
        
        # Calculate new position
        dx = linear_vel * np.cos(self.robot_orientations[robot_idx]) * dt
        dy = linear_vel * np.sin(self.robot_orientations[robot_idx]) * dt
        
        new_pos = self.robot_positions[robot_idx] + np.array([dx, dy])
        
        # Check boundaries
        new_pos = np.clip(new_pos, 0.5, self.arena_size - 0.5)
        
        # Check obstacle collision
        if not self._check_obstacle_collision(new_pos, 0.3):
            self.robot_positions[robot_idx] = new_pos
        else:
            self.collision_count += 1
        
        # Check robot-robot collision
        for j in range(self.num_robots):
            if j != robot_idx:
                dist = np.linalg.norm(self.robot_positions[robot_idx] - self.robot_positions[j])
                if dist < 0.6:
                    self.collision_count += 1
        
        self.robot_velocities[robot_idx] = np.array([linear_vel, angular_vel])
    
    def _process_packages(self):
        """Process package pickups and deliveries."""
        for i in range(self.num_robots):
            robot_pos = self.robot_positions[i]
            
            # Check for pickup
            if self.carried_packages[i] is None:
                for j, pkg in enumerate(self.packages):
                    if not pkg["picked"] and not pkg["delivered"]:
                        dist = np.linalg.norm(robot_pos - pkg["pos"])
                        if dist < 0.5:
                            pkg["picked"] = True
                            self.carried_packages[i] = j
                            break
            
            # Check for delivery
            elif self.carried_packages[i] is not None:
                pkg_idx = self.carried_packages[i]
                for zone in self.delivery_zones:
                    dist = np.linalg.norm(robot_pos - zone["pos"])
                    if dist < zone["radius"]:
                        self.packages[pkg_idx]["delivered"] = True
                        self.carried_packages[i] = None
                        self.delivered_count += 1
                        break
    
    def _calculate_reward(self) -> float:
        """Calculate team-level reward."""
        reward = 0.0
        
        # Reward for deliveries
        reward += self.delivered_count * 10.0
        
        # Penalty for collisions
        reward -= self.collision_count * 0.5
        
        # Small time penalty to encourage efficiency
        reward -= 0.01
        
        # Bonus for carrying packages (progress)
        carrying = sum(1 for c in self.carried_packages if c is not None)
        reward += carrying * 0.1
        
        return reward
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all robots."""
        observations = {}
        for i in range(self.num_robots):
            observations[f"robot_{i}"] = self._get_robot_observation(i)
        return observations
    
    def _get_robot_observation(self, robot_idx: int) -> np.ndarray:
        """Get observation for a single robot with sensor noise."""
        config = self.sensor_configs[robot_idx]
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # LiDAR observation (32 rays)
        if config["lidar_enabled"]:
            lidar = self._simulate_lidar(robot_idx, config)
            obs[:32] = lidar
        else:
            obs[:32] = 0.0  # No LiDAR
        
        # Camera features (16-dim simulated)
        if config["camera_enabled"]:
            camera = self._simulate_camera(robot_idx, config)
            obs[32:48] = camera
        else:
            obs[32:48] = 0.0  # No camera
        
        # Odometry (position + velocity with noise)
        odom_noise = config["odometry_noise"]
        obs[48:50] = self.robot_positions[robot_idx] / self.arena_size + \
                     self.np_random.normal(0, odom_noise, 2)
        obs[50:52] = self.robot_velocities[robot_idx] + \
                     self.np_random.normal(0, odom_noise, 2)
        
        # Package info (nearest package direction + distance)
        nearest_pkg = self._get_nearest_package_info(robot_idx)
        obs[52:56] = nearest_pkg
        
        return obs
    
    def _simulate_lidar(self, robot_idx: int, config: Dict) -> np.ndarray:
        """Simulate LiDAR sensor readings."""
        num_rays = 32
        max_range = config["lidar_range"]
        noise_std = config["lidar_noise"]
        
        robot_pos = self.robot_positions[robot_idx]
        robot_angle = self.robot_orientations[robot_idx]
        
        readings = np.ones(num_rays) * max_range
        
        for i in range(num_rays):
            ray_angle = robot_angle + (i / num_rays) * 2 * np.pi - np.pi
            
            # Check obstacles
            for obs in self.obstacles:
                dist = self._ray_box_intersection(
                    robot_pos, ray_angle, obs["pos"], obs["size"]
                )
                if dist is not None and dist < readings[i]:
                    readings[i] = dist
            
            # Check other robots
            for j in range(self.num_robots):
                if j != robot_idx:
                    dist = self._ray_circle_intersection(
                        robot_pos, ray_angle, self.robot_positions[j], 0.3
                    )
                    if dist is not None and dist < readings[i]:
                        readings[i] = dist
            
            # Check arena boundaries
            boundary_dist = self._ray_boundary_intersection(robot_pos, ray_angle)
            if boundary_dist < readings[i]:
                readings[i] = boundary_dist
        
        # Add noise
        readings += self.np_random.normal(0, noise_std, num_rays)
        readings = np.clip(readings, 0, max_range)
        
        # Normalize
        readings = readings / max_range
        
        return readings.astype(np.float32)
    
    def _simulate_camera(self, robot_idx: int, config: Dict) -> np.ndarray:
        """Simulate camera feature extraction."""
        noise_std = config["camera_noise"]
        
        robot_pos = self.robot_positions[robot_idx]
        robot_angle = self.robot_orientations[robot_idx]
        
        features = np.zeros(16, dtype=np.float32)
        
        # Encode visible packages in FOV
        fov = np.pi / 2  # 90 degree FOV
        visible_packages = []
        
        for pkg in self.packages:
            if not pkg["delivered"]:
                rel_pos = pkg["pos"] - robot_pos
                dist = np.linalg.norm(rel_pos)
                angle = np.arctan2(rel_pos[1], rel_pos[0]) - robot_angle
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                
                if abs(angle) < fov / 2 and dist < 8.0:
                    visible_packages.append((dist, angle, pkg["picked"]))
        
        # Encode up to 4 nearest visible packages
        visible_packages.sort(key=lambda x: x[0])
        for i, (dist, angle, picked) in enumerate(visible_packages[:4]):
            features[i*4] = dist / 8.0
            features[i*4 + 1] = angle / (np.pi / 2)
            features[i*4 + 2] = 1.0 if picked else 0.0
            features[i*4 + 3] = 1.0  # Package visible flag
        
        # Add noise
        features += self.np_random.normal(0, noise_std, 16).astype(np.float32)
        
        return features
    
    def _ray_box_intersection(self, origin: np.ndarray, angle: float, 
                               box_center: np.ndarray, box_size: np.ndarray) -> Optional[float]:
        """Calculate ray-box intersection distance."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        half_size = box_size / 2
        box_min = box_center - half_size
        box_max = box_center + half_size
        
        t_min = -np.inf
        t_max = np.inf
        
        for i in range(2):
            if abs(direction[i]) < 1e-8:
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return None
            else:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
        
        if t_min > t_max or t_max < 0:
            return None
        
        return t_min if t_min > 0 else t_max
    
    def _ray_circle_intersection(self, origin: np.ndarray, angle: float,
                                  center: np.ndarray, radius: float) -> Optional[float]:
        """Calculate ray-circle intersection distance."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        oc = origin - center
        
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        
        t = (-b + np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        
        return None
    
    def _ray_boundary_intersection(self, origin: np.ndarray, angle: float) -> float:
        """Calculate ray-boundary intersection distance."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        min_dist = float('inf')
        
        # Check all four boundaries
        boundaries = [
            (np.array([0, 0]), np.array([0, 1])),  # Left
            (np.array([self.arena_size, 0]), np.array([0, 1])),  # Right
            (np.array([0, 0]), np.array([1, 0])),  # Bottom
            (np.array([0, self.arena_size]), np.array([1, 0]))  # Top
        ]
        
        for point, normal in boundaries:
            denom = np.dot(direction, np.array([-normal[1], normal[0]]))
            if abs(denom) > 1e-8:
                t = np.dot(point - origin, np.array([-normal[1], normal[0]])) / denom
                if t > 0:
                    min_dist = min(min_dist, t)
        
        return min_dist
    
    def _get_nearest_package_info(self, robot_idx: int) -> np.ndarray:
        """Get information about the nearest unpicked package."""
        robot_pos = self.robot_positions[robot_idx]
        
        min_dist = float('inf')
        nearest_dir = np.zeros(2)
        has_package = 0.0
        
        # If carrying a package, point to nearest delivery zone
        if self.carried_packages[robot_idx] is not None:
            has_package = 1.0
            for zone in self.delivery_zones:
                dist = np.linalg.norm(robot_pos - zone["pos"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_dir = (zone["pos"] - robot_pos) / (dist + 1e-8)
        else:
            # Point to nearest unpicked package
            for pkg in self.packages:
                if not pkg["picked"] and not pkg["delivered"]:
                    dist = np.linalg.norm(robot_pos - pkg["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_dir = (pkg["pos"] - robot_pos) / (dist + 1e-8)
        
        return np.array([nearest_dir[0], nearest_dir[1], min_dist / self.arena_size, has_package], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        return {
            "delivered_count": self.delivered_count,
            "collision_count": self.collision_count,
            "step_count": self.step_count,
            "success": self.delivered_count >= self.num_packages,
            "carried_packages": sum(1 for c in self.carried_packages if c is not None)
        }
    
    def get_team_score(self) -> float:
        """Get the scalar team-level score for DC-Ada."""
        # Weighted combination matching the paper
        w_del = 10.0
        w_t = 0.01
        w_c = 0.5
        
        score = (w_del * self.delivered_count - 
                 w_t * self.step_count - 
                 w_c * self.collision_count)
        
        return score


class SearchRescueEnv(WarehouseEnv):
    """Search and Rescue variant of the environment."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault("num_packages", 12)  # "victims"
        kwargs.setdefault("max_steps", 600)
        super().__init__(**kwargs)
        self.victims_found = 0
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.victims_found = 0
        # Rename packages as victims
        for pkg in self.packages:
            pkg["found"] = False
        return obs, info
    
    def _process_packages(self):
        """Process victim finding (no delivery needed)."""
        for i in range(self.num_robots):
            robot_pos = self.robot_positions[i]
            
            for j, pkg in enumerate(self.packages):
                if not pkg.get("found", False):
                    dist = np.linalg.norm(robot_pos - pkg["pos"])
                    if dist < 1.0:  # Detection range
                        pkg["found"] = True
                        pkg["delivered"] = True  # Mark as complete
                        self.delivered_count += 1
                        self.victims_found += 1
    
    def get_team_score(self) -> float:
        """Search & Rescue score: victims found - time penalty."""
        w_vic = 10.0
        w_t = 0.01
        
        score = w_vic * self.victims_found - w_t * self.step_count
        return score


class CollaborativeMappingEnv(WarehouseEnv):
    """Collaborative Mapping variant of the environment."""
    
    def __init__(self, grid_resolution: int = 40, **kwargs):
        kwargs.setdefault("num_packages", 0)  # No packages
        kwargs.setdefault("max_steps", 800)
        super().__init__(**kwargs)
        
        self.grid_resolution = grid_resolution
        self.explored_map = None
        self.ground_truth_map = None
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        
        # Initialize maps
        self.explored_map = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        self.ground_truth_map = self._create_ground_truth_map()
        
        return obs, info
    
    def _create_ground_truth_map(self) -> np.ndarray:
        """Create ground truth occupancy map."""
        gt_map = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        cell_size = self.arena_size / self.grid_resolution
        
        for obs in self.obstacles:
            x_min = int((obs["pos"][0] - obs["size"][0]/2) / cell_size)
            x_max = int((obs["pos"][0] + obs["size"][0]/2) / cell_size)
            y_min = int((obs["pos"][1] - obs["size"][1]/2) / cell_size)
            y_max = int((obs["pos"][1] + obs["size"][1]/2) / cell_size)
            
            x_min = max(0, x_min)
            x_max = min(self.grid_resolution - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(self.grid_resolution - 1, y_max)
            
            gt_map[y_min:y_max+1, x_min:x_max+1] = 1.0
        
        return gt_map
    
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        
        # Update explored map based on robot positions and sensors
        self._update_explored_map()
        
        # Calculate IoU
        iou = self._calculate_iou()
        info["map_iou"] = iou
        
        return obs, reward, terminated, truncated, info
    
    def _update_explored_map(self):
        """Update the explored map based on robot observations."""
        cell_size = self.arena_size / self.grid_resolution
        
        for i in range(self.num_robots):
            robot_pos = self.robot_positions[i]
            config = self.sensor_configs[i]
            
            # Mark cells within sensor range as explored
            if config["lidar_enabled"]:
                sensor_range = config["lidar_range"]
            elif config["camera_enabled"]:
                sensor_range = 4.0
            else:
                sensor_range = 2.0
            
            x_center = int(robot_pos[0] / cell_size)
            y_center = int(robot_pos[1] / cell_size)
            range_cells = int(sensor_range / cell_size)
            
            for dx in range(-range_cells, range_cells + 1):
                for dy in range(-range_cells, range_cells + 1):
                    x = x_center + dx
                    y = y_center + dy
                    
                    if 0 <= x < self.grid_resolution and 0 <= y < self.grid_resolution:
                        dist = np.sqrt(dx**2 + dy**2) * cell_size
                        if dist <= sensor_range:
                            # Mark as explored with ground truth value
                            self.explored_map[y, x] = self.ground_truth_map[y, x]
    
    def _calculate_iou(self) -> float:
        """Calculate Intersection over Union of explored vs ground truth."""
        # Only consider explored cells
        explored_mask = self.explored_map > -0.5  # All explored cells
        
        if not np.any(explored_mask):
            return 0.0
        
        # For explored cells, calculate accuracy
        correct = np.sum((self.explored_map == self.ground_truth_map) & explored_mask)
        total_explored = np.sum(explored_mask)
        
        # Coverage bonus
        coverage = total_explored / (self.grid_resolution ** 2)
        
        accuracy = correct / total_explored if total_explored > 0 else 0.0
        
        # Combined IoU-like metric
        iou = accuracy * coverage
        
        return iou
    
    def get_team_score(self) -> float:
        """Mapping score: IoU - time penalty."""
        iou = self._calculate_iou()
        w_t = 0.001
        
        score = iou * 100 - w_t * self.step_count
        return score
    
    def _process_packages(self):
        """No package processing in mapping task."""
        pass


def make_env(env_type: str = "warehouse", **kwargs) -> gym.Env:
    """Factory function to create environments."""
    if env_type == "warehouse":
        return WarehouseEnv(**kwargs)
    elif env_type == "search_rescue":
        return SearchRescueEnv(**kwargs)
    elif env_type == "mapping":
        return CollaborativeMappingEnv(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


if __name__ == "__main__":
    # Test the environment
    env = make_env("warehouse", num_robots=4, seed=42)
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observations shape: {obs['robot_0'].shape}")
    
    # Run a few steps
    total_reward = 0
    for step in range(100):
        actions = {f"robot_{i}": env.action_space[f"robot_{i}"].sample() for i in range(4)}
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Completed {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Team score: {env.get_team_score():.2f}")
    print(f"Info: {info}")
