from typing import Dict, List, Optional, Text, Tuple, TypeVar

import numpy as np
import random

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import AggressiveIDMVehicle, DefensiveIDMVehicle, TruckVehicle, MotorVehicle

Observation = np.ndarray


class LLMEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "screen_width": 1920,  # [px]
                "screen_height": 270,  # [px]
                "lanes_count": 5,
                "vehicles_count": 30,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 60,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1.3,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": True,
                "aggressive_vehicle_ratio": 0.3, 
                "defensive_vehicle_ratio": 0.2,
                "truck_vehicle_ratio": 0.1,
                "motor_vehicle_ratio": 0.2,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        total_vehicles = self.config["vehicles_count"]
        num_aggressive = int(total_vehicles * self.config["aggressive_vehicle_ratio"])
        num_defensive = int(total_vehicles * self.config["defensive_vehicle_ratio"])
        num_truck = int(total_vehicles * self.config["truck_vehicle_ratio"])
        num_motor = int(total_vehicles * self.config["motor_vehicle_ratio"])
        num_normal = total_vehicles - num_aggressive - num_defensive
        self.controlled_vehicles = []

        for _ in range(total_vehicles // 2):
            self.add_random_vehicle(num_aggressive, num_defensive, num_truck, num_motor, num_normal, other_vehicles_type)

        ego_vehicle = Vehicle.create_random(
            self.road,
            speed=25,
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"]
        )
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)

        for _ in range(total_vehicles // 2, total_vehicles):
            self.add_random_vehicle(num_aggressive, num_defensive, num_truck, num_motor, num_normal, other_vehicles_type)

    def add_random_vehicle(self, num_aggressive, num_defensive, num_truck, num_motor, num_normal, other_vehicles_type):
        total_vehicles = num_aggressive + num_defensive + num_normal

        # Randomly select vehicle type based on their proportions
        rand_choice = random.randint(1, total_vehicles)
        if rand_choice <= num_aggressive:
            vehicle = AggressiveIDMVehicle.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        elif rand_choice <= num_aggressive + num_defensive:
            vehicle = DefensiveIDMVehicle.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        elif rand_choice <= num_aggressive + num_defensive + num_truck:
            vehicle = TruckVehicle.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        elif rand_choice <= num_aggressive + num_defensive + num_truck + num_motor:
            vehicle = MotorVehicle.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        else:
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])

        self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        # Existing code...
        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        if 'went_offroad' not in info:
            info['went_offroad'] = False
        if not self.vehicle.on_road:
            info['went_offroad'] = True

        if 'crash_type' not in info:
            info['crash_type'] = None
        if self.vehicle.crashed:
            crash_type = None
            for other_vehicle in self.road.vehicles:
                if other_vehicle is not self.vehicle and other_vehicle.crashed:
                    # Calculate vector to other vehicle and normalize
                    vector_to_other = other_vehicle.position
                    norm = np.linalg.norm(vector_to_other)
                    if norm != 0:
                        vector_to_other /= norm

                    # Determine relative angle
                    ego_heading_vector = [np.cos(self.vehicle.heading), np.sin(self.vehicle.heading)]
                    angle = np.arccos(np.clip(np.dot(vector_to_other, ego_heading_vector), -1, 1))

                    norm_ego = np.linalg.norm(self.vehicle.position)
                    epsilon = 5.5

                    # Determine crash type based on angle
                    if norm_ego < norm:
                        if np.abs(angle) < np.pi / 20:
                            crash_type = "front"
                        else:
                            crash_type = "front-edge"
                    elif (norm_ego > norm) and (norm_ego < norm+epsilon):
                        crash_type = "side-on"

                    else:
                        if np.abs(angle) < np.pi / 20:
                            crash_type = "rear"
                        else:
                            crash_type = "rear-edge"
                    break
            info['crash_type'] = crash_type

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def update_env_config(self, new_config):
        """
        Update the global environment configuration.

        :param new_config: A dictionary containing new configuration parameters.
        """
        self.config.update(new_config)