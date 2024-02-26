from typing import Dict, List, Optional, Text, Tuple, TypeVar

import numpy as np
import random
import ast
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import AggressiveIDMVehicle, DefensiveIDMVehicle, TruckVehicle, MotorVehicle, RegularIDMVehicle, IDMVehicle
from highway_env.envs.common.observation import TimeToCollisionObservation

Observation = np.ndarray

def calculate_speed(vehicle_class):
    params = vehicle_class.params
    speed = np.random.normal(params['speed_mean'], params['speed_std'])
    speed = max(0, speed)
    return speed

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
        total_vehicles = self.config["vehicles_count"]
        num_aggressive = int(total_vehicles * self.config["aggressive_vehicle_ratio"])
        num_defensive = int(total_vehicles * self.config["defensive_vehicle_ratio"])
        num_truck = int(total_vehicles * self.config["truck_vehicle_ratio"])
        num_normal = total_vehicles - num_aggressive - num_defensive
        self.controlled_vehicles = []

        for _ in range(total_vehicles // 2):
            self.add_random_vehicle(num_aggressive, num_defensive, num_truck, num_normal)

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
            self.add_random_vehicle(num_aggressive, num_defensive, num_truck, num_normal)


    def add_random_vehicle(self, num_aggressive, num_defensive, num_truck, num_normal):
        total_vehicles = num_aggressive + num_defensive + num_normal

        # Randomly select vehicle type based on their proportions
        rand_choice = random.randint(1, total_vehicles)
        if rand_choice <= num_aggressive:
            vehicle = AggressiveIDMVehicle.create_random(self.road, speed=calculate_speed(AggressiveIDMVehicle), spacing=1 / self.config["vehicles_density"])
        elif rand_choice <= num_aggressive + num_defensive:
            vehicle = DefensiveIDMVehicle.create_random(self.road, speed=calculate_speed(DefensiveIDMVehicle), spacing=1 / self.config["vehicles_density"])
        elif rand_choice <= num_aggressive + num_defensive + num_truck:
            vehicle = TruckVehicle.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        else:
            vehicle = RegularIDMVehicle.create_random(self.road, speed=calculate_speed(RegularIDMVehicle), spacing=1 / self.config["vehicles_density"])
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

        #new_config = ast.literal_eval(new_config)

        self.config.update(new_config)
        print("FINAL UPDATED CONFIG PLEASE DOUBLE CHECK", self.config)

    def calculate_long_and_lat_risk_indices(self, ego_vehicle: Vehicle, other_vehicle: Vehicle) -> Tuple[float, float, float]:
        # Constants from IDMVehicle settings
        rho = IDMVehicle.TIME_WANTED  # Response time of the rear vehicle
        a_max_brake = IDMVehicle.COMFORT_ACC_MIN  # Desired maximum deceleration
        distance_wanted = 1.5  # Desired jam distance to the front vehicle

        # Relative parameters for longitudinal dynamics
        v_r = ego_vehicle.speed  # Velocity of the rear vehicle (ego vehicle)
        v_f = other_vehicle.speed  # Velocity of the front vehicle
        d_lon = other_vehicle.position[0] - ego_vehicle.position[0] - other_vehicle.LENGTH  # Longitudinal distance

        # Safe longitudinal distance calculation
        d_lon_min = max(v_r * rho + distance_wanted + (v_r**2) / (2 * abs(a_max_brake)) - (v_f**2) / (2 * abs(a_max_brake)), 0)

        # Longitudinal risk index calculation
        if d_lon >= d_lon_min:
            r_lon = 0
        else:
            r_lon = 1 - d_lon / d_lon_min

        # Relative parameters for lateral dynamics
        lateral_distance_wanted = 1  # Placeholder
        lateral_velocity = abs(ego_vehicle.velocity[1])  # Lateral speed of the ego vehicle, placeholder calculation
        lateral_distance = abs(ego_vehicle.position[1] - other_vehicle.position[1])  # Lateral distance

        # Safe lateral distance calculation
        lateral_distance_safe = lateral_distance_wanted + lateral_velocity * rho

        # Lateral risk index calculation
        if lateral_distance >= lateral_distance_safe:
            r_lat = 0
        else:
            r_lat = 1 - lateral_distance / lateral_distance_safe

        return r_lon, r_lat
    
    def calculate_TTC_near_miss(self):
        self.ttc_observation = TimeToCollisionObservation(self)
        TTC_grid = self.ttc_observation.observe_new()
        TTC_grid = np.abs(TTC_grid)
        if np.any((TTC_grid < 15) & (TTC_grid > 0)):
            near_miss_occurred = True
        else:
            near_miss_occurred = False
        
        nearest_vehicles = self.ttc_observation.nearest_vehicles()

        return near_miss_occurred, nearest_vehicles

    def detect_edge_case(self, other_vehicle, near_miss_occurred):

        r_lon, r_lat = self.calculate_long_and_lat_risk_indices(self.vehicle, other_vehicle)

        threshold_lon = 2.5  # High risk of front or rear collision
        threshold_lat = 0.99  # High risk of side collision or lane departure

        # Define your edge case criteria        
        is_edge_case_lon_and_lat = r_lon > threshold_lon or r_lat > threshold_lat
        is_edge_case_TTC_near_miss = near_miss_occurred

        # print("Lat and long edge cases: ", is_edge_case_lon_and_lat)
        # print("TTC-near-miss edge case: ", is_edge_case_TTC_near_miss)

        return is_edge_case_lon_and_lat, is_edge_case_TTC_near_miss