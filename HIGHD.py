import os
import sys
import pickle
import argparse
sys.path.insert(0, '/home/zach/highD')
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from HIGHD_utils import *
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

def extract_features(tracks):
    # Initialize lists to hold feature values
    speeds = []
    accelerations = []
    lane_changes = []

    # Iterate over each track in the dataset
    for track in tracks:
        speeds.append(np.abs(np.mean(track['xVelocity'])))  # Assuming xVelocity is the speed
        accelerations.append(np.abs(np.mean(track['xAcceleration'])))  # Assuming xAcceleration is the acceleration
        lane_changes.append(sum(l1 != l2 for l1, l2 in zip(track['laneId'] , track['laneId'][1:])))

    # Create a DataFrame from the extracted features
    features = pd.DataFrame({
        'speed': speeds,
        'acceleration': accelerations,
        'lane_changes': lane_changes
    })
    return features

def perform_kprototypes_clustering(features, num_clusters=3):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=2)
    clusters = kproto.fit_predict(features, categorical=[2])  # Assuming lane_changes is at index 2
    return clusters, kproto

def visualize_clusters_3d(features, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(features) != len(labels):
        print("Mismatch in the size of features and labels.")
        return

    # Assuming features is a DataFrame and has columns 'speed', 'acceleration', and 'lane_changes'
    x = features[:, 0]  # Speed
    y = features[:, 1]  # Acceleration
    z = features[:, 2]  # Lane Changes

    ax.scatter(x, y, z, c=labels, cmap='viridis')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Acceleration')
    ax.set_zlabel('Lane Changes')
    plt.title('3D Clustering of Driving Behaviors')

    plt.show()

def load_and_process_scenario(track_number, base_dir):
    file_paths = {
        'input_path': f"{base_dir}{track_number:02}_tracks.csv",
        'input_static_path': f"{base_dir}{track_number:02}_tracksMeta.csv",
        'input_meta_path': f"{base_dir}{track_number:02}_recordingMeta.csv",
        'pickle_path': f"{base_dir}{track_number:02}.pickle"
    }

    with open(file_paths["pickle_path"], "rb") as fp:
        tracks = pickle.load(fp)
    
    features = extract_features(tracks)
    return features


def calculate_distributions(features, labels):
    cluster_distributions = {}

    # Unique cluster labels
    unique_clusters = np.unique(labels)

    # Calculate distributions for each cluster
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
        'input_path': f"{base_dir}{track_number:02}_tracks.csv",
        'input_meta_path': f"{base_dir}{track_number:02}_recordingMeta.csv",
        'pickle_path': f"{base_dir}{track_number:02}.pickle"
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

    return {
        'num_aggressive': num_aggressive,
        'num_defensive': num_defensive,
        'num_regular': num_regular,
        'num_trucks': num_trucks,
        'num_cars': num_cars
    }, end_index
if __name__ == '__main__':

    HIGHD_DIR = '/home/zach/highD/data/'
    # all_features = []

    # for i in range(1, 61):  # Assuming scenarios are numbered from 1 to 60
    #     features = load_and_process_scenario(i, HIGHD_DIR)
    #     all_features.append(features)

    # all_features_df = pd.concat(all_features, ignore_index=True)
    # all_features_df.iloc[:, 0:2] = StandardScaler().fit_transform(all_features_df.iloc[:, 0:2])

    # cluster_labels, kproto_model = perform_kprototypes_clustering(all_features_df)
    # visualize_clusters_3d(all_features_df.to_numpy(), cluster_labels)

    # distributions = calculate_distributions(all_features_df, cluster_labels_loaded)

    # for cluster, stats in distributions.items():
    #     print(f"Cluster {cluster} - Speed: Mean={stats['speed']['mean']}, Std={stats['speed']['std']}, Min={stats['speed']['min']}, Max={stats['speed']['max']}")
    #     print(f"Cluster {cluster} - Acceleration: Mean={stats['acceleration']['mean']}, Std={stats['acceleration']['std']}, Min={stats['acceleration']['min']}, Max={stats['acceleration']['max']}")
    # np.save('cluster_labels.npy', cluster_labels)
    cluster_labels = np.load('cluster_labels.npy')
    results = []
    start_index=0
    for i in range(1, 61):  
        vehicle_counts, start_index = classify_and_count_vehicles(i, HIGHD_DIR, cluster_labels, start_index)
        print(f"Scenario {i}: {vehicle_counts}")
        results.append(vehicle_counts)

    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)

    # Optionally, add a column for scenario numbers if needed
    results_df['scenario'] = range(1, len(results) + 1)

    # Save to CSV
    results_df.to_csv('highwayenv_scenario_data.csv', index=False)