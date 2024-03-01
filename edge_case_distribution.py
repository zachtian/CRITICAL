import matplotlib.pyplot as plt
import json
import numpy as np

class EdgeCaseAnalyzerFromJSON:
    def __init__(self, file_path):
        self.file_path = file_path
        self.configurations = self._load_configurations()
        self.lat_lon_counts = []
        self.ttc_near_miss_counts = []
        self._extract_edge_case_counts()

    def _load_configurations(self):
        with open(self.file_path, 'r') as json_file:
            return json.load(json_file)

    def _extract_edge_case_counts(self):
        for config in self.configurations:
            if 'edge_case_count_for_lat_and_lon' in config:
                self.lat_lon_counts.append(config['edge_case_count_for_lat_and_lon'])
            if 'edge_case_count_for_TTC_near_miss' in config:
                self.ttc_near_miss_counts.append(config['edge_case_count_for_TTC_near_miss'])

    def plot_distribution(self):
        """Generates and displays distribution plots for the edge case counts."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot and determine bins for latitude and longitude edge case counts
        n_lat_lon, bins_lat_lon, _ = axs[0].hist(self.lat_lon_counts, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
        axs[0].set_title('Distribution of Lat & Lon Edge Cases')
        axs[0].set_xlabel('Edge Case Counts')
        axs[0].set_ylabel('Frequency')

        # Plot and determine bins for TTC near miss counts
        n_ttc, bins_ttc, _ = axs[1].hist(self.ttc_near_miss_counts, bins='auto', color='orange', alpha=0.7, rwidth=0.85)
        axs[1].set_title('Distribution of TTC Near Miss Edge Cases')
        axs[1].set_xlabel('Edge Case Counts')
        axs[1].set_ylabel('Frequency')

        # plt.tight_layout()
        # plt.show()

        # Store the bins for later retrieval
        self.lat_lon_bin_edges = bins_lat_lon
        self.ttc_near_miss_bin_edges = bins_ttc

    def get_last_four_bin_values_and_frequencies(self):
        """Returns the last four bin values and their frequencies."""
        # Ensure the plot_distribution has been called to set bin edges
        if not hasattr(self, 'lat_lon_bin_edges') or not hasattr(self, 'ttc_near_miss_bin_edges'):
            raise ValueError("plot_distribution must be called before getting bin values and frequencies.")

        lat_lon_hist = np.histogram(self.lat_lon_counts, bins=self.lat_lon_bin_edges)[0]
        ttc_near_miss_hist = np.histogram(self.ttc_near_miss_counts, bins=self.ttc_near_miss_bin_edges)[0]

        # Retrieve the last four bins and their frequencies
        lat_lon_results = [(f"{self.lat_lon_bin_edges[i]} - {self.lat_lon_bin_edges[i + 1]}", lat_lon_hist[i]) for i in range(-4, 0)]
        ttc_near_miss_results = [(f"{self.ttc_near_miss_bin_edges[i]} - {self.ttc_near_miss_bin_edges[i + 1]}", ttc_near_miss_hist[i]) for i in range(-4, 0)]

        return lat_lon_results, ttc_near_miss_results
    
    def get_configurations_for_last_bins(self):
        """Returns the configurations associated with the last four bins, formatting numbers to two decimal places."""
        if not hasattr(self, 'lat_lon_bin_edges') or not hasattr(self, 'ttc_near_miss_bin_edges'):
            raise ValueError("plot_distribution must be called before getting configurations.")

        lat_lon_min_edge = self.lat_lon_bin_edges[-5]
        ttc_near_miss_min_edge = self.ttc_near_miss_bin_edges[-5]

        lat_lon_configs = [config for config in self.configurations 
                        if 'edge_case_count_for_lat_and_lon' in config and
                            lat_lon_min_edge <= config['edge_case_count_for_lat_and_lon'] < self.lat_lon_bin_edges[-1]]
        ttc_near_miss_configs = [config for config in self.configurations 
                                if 'edge_case_count_for_TTC_near_miss' in config and
                                    ttc_near_miss_min_edge <= config['edge_case_count_for_TTC_near_miss'] < self.ttc_near_miss_bin_edges[-1]]

        # Formatting with two decimal places for numeric values
        lat_lon_configs_str = ["; ".join(f"{k}: {round(v, 2) if isinstance(v, float) else v}" for k, v in config.items()) for config in lat_lon_configs]
        ttc_near_miss_configs_str = ["; ".join(f"{k}: {round(v, 2) if isinstance(v, float) else v}" for k, v in config.items()) for config in ttc_near_miss_configs]

        return lat_lon_configs_str, ttc_near_miss_configs_str