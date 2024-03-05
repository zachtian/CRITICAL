import os
import sys
import pickle
import argparse
sys.path.insert(0, '/home/zach/highD')
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from HIGHD_utils import *
from kmodes.kprototypes import KPrototypes

def perform_kprototypes_clustering(features, num_clusters=3):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=2)
    clusters = kproto.fit_predict(features, categorical=[2])  # Assuming lane_changes is at index 2
    return clusters, kproto

def visualize_clusters_3d(features, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = features[:, 0]  
    y = features[:, 1]  
    z = features[:, 2]  

    ax.scatter(x, y, z, c=labels, cmap='viridis')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Acceleration')
    ax.set_zlabel('Lane Changes')
    plt.title('3D Clustering of Driving Behaviors')

    plt.show()

def find_most_dangerous_scenario(tracks):
    most_dangerous_pair = None
    highest_risk_score = -1
    time_of_event = None

    for track_i in tracks:
        relevant_keys = ['precedingId', 'followingId']

        # Retrieve and flatten relevant IDs, then add to a set for uniqueness
        relevant_ids_set = set()
        for key in relevant_keys:
            if track_i.get(key) is not None:
                relevant_ids = np.array(track_i[key]).flatten()
                relevant_ids_set.update(relevant_ids)
        relevant_ids_set.discard(0)
        for rel_id in relevant_ids_set:
            track_j = tracks[rel_id-1]

            set_a = set(track_i['frame'])
            set_b = set(track_j['frame'])
            common_frames = list(set_a.intersection(set_b))
            common_frames.sort() 

            for frame in common_frames:
                frame_index_i = np.where(track_i['frame'] == frame)[0][0]
                frame_index_j = np.where(track_j['frame'] == frame)[0][0]
                if track_i['ttc'][frame_index_i] != 0 and track_j['ttc'][frame_index_j] != 0:
                # Calculate risk score only if both TTC values are positive
                    ttc_i = min(np.abs(track_i['ttc'][frame_index_i]), np.abs(track_j['ttc'][frame_index_j]))
                    thw_i = min(track_i['thw'][frame_index_i], track_j['thw'][frame_index_j])
                    # dhw_i = max(track_i['dhw'][frame_index_i], track_j['dhw'][frame_index_j])

                    risk_score = (1 / (ttc_i + 1e-5)) + (4 / (thw_i + 1e-5)) 

                    if risk_score > highest_risk_score:
                        highest_risk_score = risk_score
                        vehicle_i_info = {
                            'location': track_i['bbox'][frame_index_i],
                            'speed': track_i['xVelocity'][frame_index_i],
                            'acceleration': track_i['xAcceleration'][frame_index_i],
                            'lane_id': track_i['laneId'][frame_index_i]
                        }

                        vehicle_j_info = {
                            'location': track_j['bbox'][frame_index_j],
                            'speed': track_j['xVelocity'][frame_index_j],
                            'acceleration': track_j['xAcceleration'][frame_index_j],
                            'lane_id': track_j['laneId'][frame_index_j]
                        }


    return vehicle_i_info, vehicle_j_info

def load_and_process_scenario(track_number, base_dir):
    file_paths = {
        'input_static_path': f"{base_dir}{track_number:02}_tracksMeta.csv",
        'pickle_path': f"{base_dir}{track_number:02}.pickle",
        'input_meta_path' : f"{base_dir}{track_number:02}_recordingMeta.csv"
    }
    with open(file_paths["pickle_path"], "rb") as fp:
        tracks = pickle.load(fp)
    with open(file_paths["input_meta_path"], "rb") as fp:
        static_info = read_static_info(file_paths)

    car_speeds = []
    car_accelerations = []
    car_lane_changes = []
    
    truck_speeds = []
    truck_accelerations = []
    vehicle_i_info, vehicle_j_info = find_most_dangerous_scenario(tracks)

    for track in tracks:
        if type(track['id']) ==np.ndarray:
            v_id = track['id'][0]
        else:
            v_id = track['id']
        info = static_info[v_id]
        if info['class'] == 'Car':
            car_speeds.append(np.abs(info['meanXVelocity']))
            car_accelerations.append(np.abs(np.mean(track['xAcceleration'])))
            car_lane_changes.append(info['numLaneChanges'])
        elif info['class'] == 'Truck':
            truck_speeds.append(np.abs(info['meanXVelocity']))
            truck_accelerations.append(np.abs(np.mean(track['xAcceleration'])))

    car_features = pd.DataFrame({
        'speed': car_speeds,
        'acceleration': car_accelerations,
        'lane_changes': car_lane_changes
    })

    truck_features = pd.DataFrame({
        'speed': truck_speeds,
        'acceleration': truck_accelerations
    })

    return vehicle_i_info, vehicle_j_info, car_features, truck_features

def calculate_distributions(features, labels):
    cluster_distributions = {}

    unique_clusters = np.unique(labels)

    for cluster in unique_clusters:
        cluster_data = features[labels == cluster]
        speed_stats = {
            'mean': np.mean(cluster_data['speed']),
            'std': np.std(cluster_data['speed']),
            'min': np.min(cluster_data['speed']),
            'max': np.max(cluster_data['speed'])
        }
        acceleration_stats = {
            'mean': np.mean(cluster_data['acceleration']),
            'std': np.std(cluster_data['acceleration']),
            'min': np.min(cluster_data['acceleration']),
            'max': np.max(cluster_data['acceleration'])
        }
        cluster_distributions[cluster] = {'speed': speed_stats, 'acceleration': acceleration_stats}

    return cluster_distributions

def classify_and_count_vehicles(track_number, base_dir, cluster_labels, start_index):
    file_paths = {
        'input_static_path': f"{base_dir}{track_number:02}_tracksMeta.csv",
        'pickle_path': f"{base_dir}{track_number:02}.pickle",
        'input_meta_path' : f"{base_dir}{track_number:02}_recordingMeta.csv"
    }

    with open(file_paths["pickle_path"], "rb") as fp:
        tracks = pickle.load(fp)
    with open(file_paths["input_meta_path"], "rb") as fp:
        meta = read_meta_info(file_paths)
    features = extract_features(tracks)
    end_index = start_index + len(tracks)
    scenario_labels = cluster_labels[start_index:end_index]

    num_aggressive = sum(scenario_labels == 1)
    num_defensive = sum(scenario_labels == 2)
    num_regular = sum(scenario_labels == 0)

    num_trucks = meta['numTrucks']
    num_cars = meta['numCars']
    total_vehicles = num_aggressive + num_defensive + num_regular

    density = num_vehicles / 420

    return {
        'num_aggressive': num_aggressive,
        'num_defensive': num_defensive,
        'num_regular': num_regular,
        'num_trucks': num_trucks,
        'num_cars': num_cars,
        'density': density 
    }, end_index

if __name__ == '__main__':

    HIGHD_DIR = '/home/zach/highD/data/'
    all_car_features = []
    all_truck_features = []
    dangerous_scenarios = {}
    for i in range(1, 61):  # Assuming scenarios are numbered from 1 to 60
        vehicle_i_info, vehicle_j_info, car_features, truck_features = load_and_process_scenario(i, HIGHD_DIR)
        all_car_features.append(car_features)
        all_truck_features.append(truck_features)
        dangerous_scenarios[i] = {
            'vehicle_i_info': vehicle_i_info,
            'vehicle_j_info': vehicle_j_info,
            'scenario': i
        }
    scenarios_df = pd.DataFrame.from_dict(dangerous_scenarios, orient='index')

    scenarios_df.to_csv('highwayenv_scenario_data.csv', index=False)
    all_car_features_df = pd.concat(all_car_features, ignore_index=True)
    all_car_features_df_scaled = all_car_features_df.copy()
    scaler = MinMaxScaler()
    all_car_features_df_scaled = scaler.fit_transform(all_car_features_df[['speed', 'acceleration']])
    all_car_features_df_scaled = pd.DataFrame(all_car_features_df_scaled, columns=['speed', 'acceleration'])

    all_car_features_df_scaled['lane_changes'] = all_car_features_df['lane_changes']
    cluster_labels, kproto_model = perform_kprototypes_clustering(all_car_features_df_scaled)
    visualize_clusters_3d(all_car_features_df_scaled.to_numpy(), cluster_labels)

    distributions = calculate_distributions(all_car_features_df_scaled, cluster_labels)

    for cluster, stats in distributions.items():
        print(f"Cluster {cluster} - Speed: Mean={stats['speed']['mean']}, Std={stats['speed']['std']}, Min={stats['speed']['min']}, Max={stats['speed']['max']}")
        print(f"Cluster {cluster} - Acceleration: Mean={stats['acceleration']['mean']}, Std={stats['acceleration']['std']}, Min={stats['acceleration']['min']}, Max={stats['acceleration']['max']}")
    np.save('cluster_labels.npy', cluster_labels)

    # Concatenate all truck features into a single DataFrame
    all_truck_features_df = pd.concat(all_truck_features, ignore_index=True)

    # Calculate statistics for trucks
    truck_speed_stats = {
        'mean': np.mean(all_truck_features_df['speed']),
        'std': np.std(all_truck_features_df['speed']),
        'min': np.min(all_truck_features_df['speed']),
        'max': np.max(all_truck_features_df['speed'])
    }
    truck_acceleration_stats = {
        'mean': np.mean(all_truck_features_df['acceleration']),
        'std': np.std(all_truck_features_df['acceleration']),
        'min': np.min(all_truck_features_df['acceleration']),
        'max': np.max(all_truck_features_df['acceleration'])
    }

    print(f"Truck Speed: Mean={truck_speed_stats['mean']}, Std={truck_speed_stats['std']}, Min={truck_speed_stats['min']}, Max={truck_speed_stats['max']}")
    print(f"Truck Acceleration: Mean={truck_acceleration_stats['mean']}, Std={truck_acceleration_stats['std']}, Min={truck_acceleration_stats['min']}, Max={truck_acceleration_stats['max']}")
    # cluster_labels = np.load('cluster_labels.npy')
    # results = []
    # start_index=0
    # for i in range(1, 61):  
    #     vehicle_counts, start_index = classify_and_count_vehicles(i, HIGHD_DIR, cluster_labels, start_index)
    #     print(f"Scenario {i}: {vehicle_counts}")
    #     results.append(vehicle_counts)

    # # Convert list of dictionaries to DataFrame
    # results_df = pd.DataFrame(results)

    # # Optionally, add a column for scenario numbers if needed
    # results_df['scenario'] = range(1, len(results) + 1)

    # # Save to CSV
    # results_df.to_csv('highwayenv_scenario_data.csv', index=False)