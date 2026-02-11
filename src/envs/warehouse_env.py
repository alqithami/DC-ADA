"""
Warehouse Environment - Pure NumPy Implementation (No PyBullet)

Multi-robot warehouse logistics environment where robots must:
1. Navigate to pickup locations
2. Collect packages
3. Deliver packages to drop-off zones
4. Avoid collisions with obstacles and other robots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_env import BaseMultiRobotEnv, Robot


class WarehouseEnv(BaseMultiRobotEnv):
    """
    Warehouse logistics environment for multi-robot coordination.
    
    Robots must pick up packages and deliver them to designated zones.
    Different robots have different sensor configurations (heterogeneity).
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        heterogeneity_level: int = 1,
        num_packages: int = 8,
        num_dropoff_zones: int = 2,
        world_size: Tuple[float, float] = (20.0, 20.0),
        max_steps: int = 500,
        seed: Optional[int] = None,
        # Reward weights / success threshold
        delivery_reward: float = 100.0,
        pickup_reward: float = 1.0,
        collision_penalty: float = 10.0,
        time_penalty: float = 0.1,
        progress_weight: float = 0.01,
        progress_scale: float = 20.0,
        target_deliveries: Optional[int] = None,
    ):
        self.num_packages = num_packages
        self.num_dropoff_zones = num_dropoff_zones

        # Reward configuration
        self.delivery_reward = float(delivery_reward)
        self.pickup_reward = float(pickup_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        self.progress_weight = float(progress_weight)
        self.progress_scale = float(progress_scale)
        # Success threshold: how many deliveries constitute a "successful" episode.
        # Default is 50% of packages (rounded up), which yields a non-trivial but
        # achievable success signal.
        if target_deliveries is None:
            self.target_deliveries = max(1, int(np.ceil(num_packages * 0.5)))
        else:
            self.target_deliveries = int(target_deliveries)

        # LiDAR will also detect packages as small circles (handled in BaseMultiRobotEnv).
        # This keeps the sensor heterogeneity meaningful (e.g., LiDAR-only robots can
        # still perceive packages).
        self.lidar_detects_targets = True
        self.lidar_target_radius = 0.35
        
        # Package state
        self.packages: List[Dict] = []
        self.dropoff_zones: List[np.ndarray] = []
        self.robot_carrying: List[Optional[int]] = []  # Which package each robot carries
        
        # Metrics
        self.delivered_count = 0
        # Collision_count is the cumulative number of *collision events* (not a per-step accumulating penalty)
        self.collision_count = 0
        self.picked_up_count = 0

        # Bookkeeping for per-step delta rewards
        self._prev_delivered_count = 0
        self._prev_collision_count = 0
        self._prev_picked_up_count = 0
        self._prev_colliding_pairs = set()
        
        super().__init__(
            num_robots=num_robots,
            heterogeneity_level=heterogeneity_level,
            world_size=world_size,
            max_steps=max_steps,
            seed=seed
        )
        
    def _init_environment(self):
        """Initialize warehouse-specific elements."""
        # Reset metrics
        self.delivered_count = 0
        self.collision_count = 0
        self.picked_up_count = 0
        self.robot_carrying = [None] * self.num_robots

        # Reset delta trackers
        self._prev_delivered_count = 0
        self._prev_collision_count = 0
        self._prev_picked_up_count = 0
        self._prev_colliding_pairs = set()
        
        # Create obstacles (shelving units)
        self.obstacles = []
        shelf_positions = [
            (5, 5), (5, 15), (15, 5), (15, 15),
            (10, 8), (10, 12)
        ]
        for pos in shelf_positions:
            if pos[0] < self.world_size[0] and pos[1] < self.world_size[1]:
                self.obstacles.append(np.array([pos[0], pos[1], 1.0]))  # x, y, radius
        
        # Create packages at random locations
        self.packages = []
        for i in range(self.num_packages):
            while True:
                pos = np.array([
                    self.rng.uniform(2, self.world_size[0] - 2),
                    self.rng.uniform(2, self.world_size[1] - 2)
                ])
                # Check not too close to obstacles
                valid = True
                for obs in self.obstacles:
                    if np.linalg.norm(pos - obs[:2]) < 2.0:
                        valid = False
                        break
                if valid:
                    break
            
            self.packages.append({
                'id': i,
                'position': pos,
                'picked_up': False,
                'delivered': False
            })
        
        # Create drop-off zones
        self.dropoff_zones = [
            np.array([2.0, self.world_size[1] / 2]),
            np.array([self.world_size[0] - 2.0, self.world_size[1] / 2])
        ]
        
        # Set targets for observation (package positions)
        self.targets = [p['position'] for p in self.packages if not p['delivered']]
        
    def _update_environment(self):
        """Update package pickups and deliveries."""
        pickup_radius = 1.0
        dropoff_radius = 2.0
        
        for robot_idx, robot in enumerate(self.robots):
            # Check for package pickup
            if self.robot_carrying[robot_idx] is None:
                for pkg in self.packages:
                    if not pkg['picked_up'] and not pkg['delivered']:
                        dist = np.linalg.norm(robot.position - pkg['position'])
                        if dist < pickup_radius:
                            pkg['picked_up'] = True
                            self.robot_carrying[robot_idx] = pkg['id']
                            self.picked_up_count += 1
                            break
            
            # Check for package delivery
            elif self.robot_carrying[robot_idx] is not None:
                for dropoff in self.dropoff_zones:
                    dist = np.linalg.norm(robot.position - dropoff)
                    if dist < dropoff_radius:
                        pkg_id = self.robot_carrying[robot_idx]
                        self.packages[pkg_id]['delivered'] = True
                        self.robot_carrying[robot_idx] = None
                        self.delivered_count += 1
                        break
        
        # Check robot-robot collisions (count *collision events*, not sustained contact)
        collision_radius = 0.5
        current_pairs = set()
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                dist = np.linalg.norm(self.robots[i].position - self.robots[j].position)
                if dist < collision_radius:
                    current_pairs.add((i, j))

        new_pairs = current_pairs - self._prev_colliding_pairs
        if new_pairs:
            self.collision_count += len(new_pairs)
        self._prev_colliding_pairs = current_pairs
        
        # Update targets list
        self.targets = [p['position'] for p in self.packages if not p['delivered'] and not p['picked_up']]
        
    def _compute_reward(self) -> float:
        """
        Compute scalar team reward.
        
        Reward structure:
        - +100 for each package delivered
        - -10 for robot-robot collision
        - -0.1 per timestep (encourage efficiency)
        - +1 for picking up a package
        """
        reward = 0.0

        # Time penalty (encourage efficiency)
        reward -= self.time_penalty

        # Event-based deltas (prevents reward explosion)
        delivered_delta = self.delivered_count - self._prev_delivered_count
        collision_delta = self.collision_count - self._prev_collision_count
        pickup_delta = self.picked_up_count - self._prev_picked_up_count

        # Update trackers
        self._prev_delivered_count = self.delivered_count
        self._prev_collision_count = self.collision_count
        self._prev_picked_up_count = self.picked_up_count

        # Delivery / pickup / collision shaping
        reward += delivered_delta * self.delivery_reward
        reward += pickup_delta * self.pickup_reward
        reward -= collision_delta * self.collision_penalty

        # Progress shaping:
        #  - if not carrying: move toward the nearest available package
        #  - if carrying: move toward the nearest drop-off zone
        for robot_idx, robot in enumerate(self.robots):
            min_dist = float('inf')

            if self.robot_carrying[robot_idx] is None:
                # Seek packages
                for pkg in self.packages:
                    if not pkg['delivered'] and not pkg['picked_up']:
                        dist = np.linalg.norm(robot.position - pkg['position'])
                        min_dist = min(min_dist, dist)
            else:
                # Seek drop-off
                for dropoff in self.dropoff_zones:
                    dist = np.linalg.norm(robot.position - dropoff)
                    min_dist = min(min_dist, dist)

            if min_dist < float('inf'):
                reward += self.progress_weight * (self.progress_scale - min_dist)  # Closer is better
        
        return float(reward)
    
    def _check_done(self) -> bool:
        """Check if episode is complete."""
        # Done if all packages delivered
        if self.delivered_count >= self.target_deliveries:
            return True
        
        # Done if max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """Get environment information."""
        packages_remaining = sum(1 for p in self.packages if not p['delivered'])
        success = self.delivered_count >= self.target_deliveries
        delivery_ratio = float(self.delivered_count) / float(max(1, self.target_deliveries))
        
        return {
            'delivered_count': self.delivered_count,
            'target_deliveries': self.target_deliveries,
            'delivery_ratio': delivery_ratio,
            'packages_remaining': packages_remaining,
            'collision_count': self.collision_count,
            'picked_up_count': self.picked_up_count,
            'success': success,
            'step_count': self.step_count
        }

    # ------------------------------------------------------------------
    # Task-level observation features
    # ------------------------------------------------------------------
    def _get_extra_obs_dim(self) -> int:
        """Extra features appended to every robot observation.

        Features:
          1) carrying flag (0/1)
          2) relative vectors to each drop-off zone (dx, dy per zone)
        """
        return 1 + 2 * len(self.dropoff_zones)

    def _get_extra_obs(self, robot: Robot) -> np.ndarray:
        rid = int(robot.robot_id)
        carrying = 1.0 if self.robot_carrying[rid] is not None else 0.0
        ws = np.asarray(self.world_size, dtype=np.float32)
        rels: List[float] = []
        for drop in self.dropoff_zones:
            rel = (np.asarray(drop, dtype=np.float32) - robot.position) / ws
            rels.extend([float(rel[0]), float(rel[1])])
        return np.asarray([carrying, *rels], dtype=np.float32)

class SearchRescueEnv(BaseMultiRobotEnv):
    """
    Search and Rescue environment for multi-robot coordination.
    
    Robots must locate and rescue victims in a disaster scenario.
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        heterogeneity_level: int = 1,
        num_victims: int = 6,
        world_size: Tuple[float, float] = (30.0, 30.0),
        max_steps: int = 500,
        seed: Optional[int] = None,
        # Reward weights / success threshold
        found_reward: float = 5.0,
        rescue_reward: float = 20.0,
        time_penalty: float = 0.1,
        coverage_reward: float = 0.5,
        rescue_health_bonus_scale: float = 0.1,
        target_rescues: Optional[int] = None,
    ):
        self.num_victims = num_victims
        self.victims: List[Dict] = []
        self.rescued_count = 0
        self.found_count = 0

        # Reward configuration
        self.found_reward = float(found_reward)
        self.rescue_reward = float(rescue_reward)
        self.time_penalty = float(time_penalty)
        self.coverage_reward = float(coverage_reward)
        self.rescue_health_bonus_scale = float(rescue_health_bonus_scale)
        # Success threshold: how many rescues constitute a "successful" episode.
        # Default is 50% of victims (rounded up), which yields a non-trivial but
        # achievable success signal.
        if target_rescues is None:
            self.target_rescues = max(1, int(np.ceil(num_victims * 0.5)))
        else:
            self.target_rescues = int(target_rescues)

        # LiDAR will also detect victims as small circles (handled in BaseMultiRobotEnv).
        self.lidar_detects_targets = True
        self.lidar_target_radius = 0.35

        # Delta bookkeeping
        self._prev_found_count = 0
        self._prev_rescued_count = 0
        self._health_bonus_this_step = 0.0
        
        super().__init__(
            num_robots=num_robots,
            heterogeneity_level=heterogeneity_level,
            world_size=world_size,
            max_steps=max_steps,
            seed=seed
        )
        
    def _init_environment(self):
        """Initialize search and rescue scenario."""
        self.rescued_count = 0
        self.found_count = 0

        # Reset delta trackers
        self._prev_found_count = 0
        self._prev_rescued_count = 0
        self._health_bonus_this_step = 0.0
        
        # Create debris obstacles
        self.obstacles = []
        num_debris = 15
        for _ in range(num_debris):
            pos = np.array([
                self.rng.uniform(3, self.world_size[0] - 3),
                self.rng.uniform(3, self.world_size[1] - 3)
            ])
            radius = self.rng.uniform(0.5, 1.5)
            self.obstacles.append(np.array([pos[0], pos[1], radius]))
        
        # Create victims at random locations
        self.victims = []
        for i in range(self.num_victims):
            while True:
                pos = np.array([
                    self.rng.uniform(2, self.world_size[0] - 2),
                    self.rng.uniform(2, self.world_size[1] - 2)
                ])
                valid = True
                for obs in self.obstacles:
                    if np.linalg.norm(pos - obs[:2]) < obs[2] + 1.0:
                        valid = False
                        break
                if valid:
                    break
            
            self.victims.append({
                'id': i,
                'position': pos,
                'found': False,
                'rescued': False,
                'health': 100.0  # Decreases over time
            })
        
        self.targets = [v['position'] for v in self.victims]
        
    def _update_environment(self):
        """Update victim states and rescues."""
        detection_radius = 3.0
        rescue_radius = 1.0

        # Reset per-step health bonus (only awarded on rescue event)
        self._health_bonus_this_step = 0.0
        
        # Decrease victim health over time
        for victim in self.victims:
            if not victim['rescued']:
                victim['health'] -= 0.1
        
        for robot in self.robots:
            for victim in self.victims:
                if victim['rescued']:
                    continue
                    
                dist = np.linalg.norm(robot.position - victim['position'])
                
                # Detection
                if not victim['found'] and dist < detection_radius:
                    victim['found'] = True
                    self.found_count += 1
                
                # Rescue
                if victim['found'] and dist < rescue_radius:
                    victim['rescued'] = True
                    self.rescued_count += 1
                    # Reward faster rescues by crediting the remaining health at rescue time
                    self._health_bonus_this_step += float(victim['health']) * self.rescue_health_bonus_scale
        
        self.targets = [v['position'] for v in self.victims if not v['rescued']]
        
    def _compute_reward(self) -> float:
        """Compute reward based on rescue progress.

        IMPORTANT: Use *delta-based* event rewards (found/rescued) rather than
        multiplying cumulative counters at every step. This keeps reward scale
        stable and prevents inadvertent reward explosion.
        """
        reward = 0.0

        # Time penalty
        reward -= self.time_penalty

        # Event-based deltas
        found_delta = self.found_count - self._prev_found_count
        rescued_delta = self.rescued_count - self._prev_rescued_count

        self._prev_found_count = self.found_count
        self._prev_rescued_count = self.rescued_count

        reward += found_delta * self.found_reward
        reward += rescued_delta * self.rescue_reward
        reward += float(self._health_bonus_this_step)

        # Exploration bonus
        coverage = self._compute_coverage()
        reward += coverage * self.coverage_reward

        return float(reward)
    
    def _compute_coverage(self) -> float:
        """Compute area coverage by robots."""
        # Simple grid-based coverage
        grid_size = 5
        grid = np.zeros((grid_size, grid_size))
        
        for robot in self.robots:
            gx = int(robot.position[0] / self.world_size[0] * grid_size)
            gy = int(robot.position[1] / self.world_size[1] * grid_size)
            gx = np.clip(gx, 0, grid_size - 1)
            gy = np.clip(gy, 0, grid_size - 1)
            grid[gx, gy] = 1.0
        
        return np.sum(grid) / (grid_size * grid_size)
    
    def _check_done(self) -> bool:
        """Check if episode is complete."""
        if self.rescued_count >= self.target_rescues:
            return True
        if self.step_count >= self.max_steps:
            return True
        # All victims dead
        if all(v['health'] <= 0 or v['rescued'] for v in self.victims):
            return True
        return False
    
    def _get_info(self) -> Dict:
        """Get environment information."""
        rescue_ratio = float(self.rescued_count) / float(max(1, self.target_rescues))
        found_ratio = float(self.found_count) / float(max(1, self.num_victims))

        return {
            'found_count': self.found_count,
            'found_ratio': found_ratio,
            'rescued_count': self.rescued_count,
            'target_rescues': self.target_rescues,
            'rescue_ratio': rescue_ratio,
            'victims_remaining': self.num_victims - self.rescued_count,
            'success': self.rescued_count >= self.target_rescues,
            'step_count': self.step_count
        }

    # ------------------------------------------------------------------
    # Task-level observation features
    # ------------------------------------------------------------------
    def _get_extra_obs_dim(self) -> int:
        """Extra features appended to every robot observation.

        Features (all normalized):
          1) found_ratio   = found_count / num_victims
          2) rescued_ratio = rescued_count / num_victims
          3) time_frac     = step_count / max_steps
        """
        return 3

    def _get_extra_obs(self, robot: Robot) -> np.ndarray:
        denom = float(max(1, self.num_victims))
        found_ratio = float(self.found_count) / denom
        rescued_ratio = float(self.rescued_count) / denom
        time_frac = float(self.step_count) / float(max(1, self.max_steps))
        return np.asarray([found_ratio, rescued_ratio, time_frac], dtype=np.float32)


class CollaborativeMappingEnv(BaseMultiRobotEnv):
    """
    Collaborative Mapping environment for multi-robot coordination.
    
    Robots must explore and map an unknown environment together.
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        heterogeneity_level: int = 1,
        grid_resolution: int = 20,
        world_size: Tuple[float, float] = (20.0, 20.0),
        target_coverage: float = 0.8,
        max_steps: int = 500,
        seed: Optional[int] = None
    ):
        self.grid_resolution = grid_resolution
        self.exploration_map: Optional[np.ndarray] = None
        self.coverage_history: List[float] = []

        # Success threshold (coverage ratio)
        self.target_coverage = float(target_coverage)

        # LiDAR can detect points of interest targets as small circles (optional)
        self.lidar_detects_targets = True
        self.lidar_target_radius = 0.35
        
        super().__init__(
            num_robots=num_robots,
            heterogeneity_level=heterogeneity_level,
            world_size=world_size,
            max_steps=max_steps,
            seed=seed
        )
        
    def _init_environment(self):
        """Initialize mapping scenario."""
        # Exploration map (0 = unexplored, 1 = explored)
        self.exploration_map = np.zeros((self.grid_resolution, self.grid_resolution))
        self.coverage_history = []
        
        # Create random obstacles
        self.obstacles = []
        num_obstacles = 10
        for _ in range(num_obstacles):
            pos = np.array([
                self.rng.uniform(2, self.world_size[0] - 2),
                self.rng.uniform(2, self.world_size[1] - 2)
            ])
            radius = self.rng.uniform(0.5, 1.5)
            self.obstacles.append(np.array([pos[0], pos[1], radius]))
        
        # Points of interest for observation targets
        self.targets = []
        for _ in range(8):
            pos = np.array([
                self.rng.uniform(2, self.world_size[0] - 2),
                self.rng.uniform(2, self.world_size[1] - 2)
            ])
            self.targets.append(pos)
        
    def _update_environment(self):
        """Update exploration map based on robot positions."""
        sensor_range = 3.0
        
        for robot in self.robots:
            # Mark cells within sensor range as explored
            gx_center = int(robot.position[0] / self.world_size[0] * self.grid_resolution)
            gy_center = int(robot.position[1] / self.world_size[1] * self.grid_resolution)
            
            range_cells = int(sensor_range / self.world_size[0] * self.grid_resolution)
            
            for dx in range(-range_cells, range_cells + 1):
                for dy in range(-range_cells, range_cells + 1):
                    gx = gx_center + dx
                    gy = gy_center + dy
                    if 0 <= gx < self.grid_resolution and 0 <= gy < self.grid_resolution:
                        # Check line of sight (simple version)
                        self.exploration_map[gx, gy] = 1.0
        
        # Record coverage
        coverage = np.sum(self.exploration_map) / (self.grid_resolution ** 2)
        self.coverage_history.append(coverage)
        
    def _compute_reward(self) -> float:
        """Compute reward based on exploration progress."""
        reward = 0.0
        
        # Time penalty
        reward -= 0.1
        
        # Coverage reward
        coverage = np.sum(self.exploration_map) / (self.grid_resolution ** 2)
        reward += coverage * 10.0
        
        # New area discovered bonus
        if len(self.coverage_history) >= 2:
            coverage_delta = self.coverage_history[-1] - self.coverage_history[-2]
            reward += coverage_delta * 50.0
        
        # Spread bonus (encourage robots to spread out)
        spread = self._compute_spread()
        reward += spread * 0.5
        
        return float(reward)
    
    def _compute_spread(self) -> float:
        """Compute how spread out the robots are."""
        if len(self.robots) < 2:
            return 0.0
        
        positions = np.array([r.position for r in self.robots])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)
    
    def _check_done(self) -> bool:
        """Check if episode is complete."""
        coverage = np.sum(self.exploration_map) / (self.grid_resolution ** 2)
        
        if coverage >= self.target_coverage:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False
    
    def _get_info(self) -> Dict:
        """Get environment information."""
        coverage = np.sum(self.exploration_map) / (self.grid_resolution ** 2)
        return {
            'coverage': coverage,
            'target_coverage': self.target_coverage,
            'success': coverage >= self.target_coverage,
            'step_count': self.step_count
        }

    # ------------------------------------------------------------------
    # Task-level observation features
    # ------------------------------------------------------------------
    def _get_extra_obs_dim(self) -> int:
        """Extra features appended to every robot observation.

        Features:
          1) current coverage (0..1)
          2) time_frac = step_count / max_steps
        """
        return 2

    def _get_extra_obs(self, robot: Robot) -> np.ndarray:
        coverage = float(np.sum(self.exploration_map) / (self.grid_resolution ** 2))
        time_frac = float(self.step_count) / float(max(1, self.max_steps))
        return np.asarray([coverage, time_frac], dtype=np.float32)
