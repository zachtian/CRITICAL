import matplotlib.pyplot as plt
import json
import numpy as np
import os

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
        
    # def _extract_edge_case_counts(self):
    #     for config in self.configurations:
    #         if 'edge_case_count_for_lat_and_lon' in config:
    #             self.lat_lon_counts.append(config['edge_case_count_for_lat_and_lon'])
    #         if 'edge_case_count_for_TTC_near_miss' in config:
    #             self.ttc_near_miss_counts.append(config['edge_case_count_for_TTC_near_miss'])

    def _extract_edge_case_counts(self):
        # Determine the start index for the last 20% of the configurations
        start_idx = int(len(self.configurations) * 0)
        end_idx = int(len(self.configurations) * 1)
        for config in self.configurations[start_idx:end_idx]:
            if 'edge_case_count_for_lat_and_lon' in config:
                self.lat_lon_counts.append(config['edge_case_count_for_lat_and_lon'])
            if 'edge_case_count_for_TTC_near_miss' in config:
                self.ttc_near_miss_counts.append(config['edge_case_count_for_TTC_near_miss'])

    # def plot_distribution(self):
    #     """Generates and displays distribution plots for the edge case counts."""
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #     # Plot and determine bins for latitude and longitude edge case counts
    #     n_lat_lon, bins_lat_lon, _ = axs[0].hist(self.lat_lon_counts, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    #     axs[0].set_title('Distribution of Lat & Lon Edge Cases')
    #     axs[0].set_xlabel('Edge Case Counts')
    #     axs[0].set_ylabel('Frequency')

    #     # Plot and determine bins for TTC near miss counts
    #     n_ttc, bins_ttc, _ = axs[1].hist(self.ttc_near_miss_counts, bins='auto', color='orange', alpha=0.7, rwidth=0.85)
    #     axs[1].set_title('Distribution of TTC Near Miss Edge Cases')
    #     axs[1].set_xlabel('Edge Case Counts')
    #     axs[1].set_ylabel('Frequency')

    #     plt.tight_layout()
    #     plt.show()

    #     # Store the bins for later retrieval
    #     self.lat_lon_bin_edges = bins_lat_lon
    #     self.ttc_near_miss_bin_edges = bins_ttc

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
    
    def _plot_distributions(self, save_path):
        """Generates and saves distribution plots for the edge case counts."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 3))
        axs[0].hist(self.lat_lon_counts, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
        axs[0].set_title('Distribution of Lat & Lon Edge Cases')
        axs[0].set_xlabel('Edge Case Counts')
        axs[0].set_ylabel('Frequency')
        axs[1].hist(self.ttc_near_miss_counts, bins='auto', color='orange', alpha=0.7, rwidth=0.85)
        axs[1].set_title('Distribution of TTC Near Miss Edge Cases')
        axs[1].set_xlabel('Edge Case Counts')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_distribution_info(self, base_dir, subdir):
        """Generates and displays distribution plots with properly aligned annotations and updated color schemes."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))

        # Compute statistics for latitude and longitude
        lat_lon_stats = {
            'Mean': np.mean(self.lat_lon_counts),
            'Variance': np.var(self.lat_lon_counts),
            'Std. Dev.': np.std(self.lat_lon_counts),
            'Min': np.min(self.lat_lon_counts),
            'Max': np.max(self.lat_lon_counts)
        }

        # Compute statistics for TTC near miss
        ttc_stats = {
            'Mean': np.mean(self.ttc_near_miss_counts),
            'Variance': np.var(self.ttc_near_miss_counts),
            'Std. Dev.': np.std(self.ttc_near_miss_counts),
            'Min': np.min(self.ttc_near_miss_counts),
            'Max': np.max(self.ttc_near_miss_counts)
        }

        # Define a function to format and color the text boxes for stats
        def add_aligned_stats_box(ax, stats, text_color, box_color):
            max_label_length = max(len(label) for label in stats.keys()) + 1  # +1 for the colon
            max_value_length = max(len(f"{value:.2f}") for value in stats.values())
            formatted_text = '\n'.join(f'{label}:' + ' ' * (max_label_length - len(label)) + f'{value:>{max_value_length}.2f}'
                                       for label, value in stats.items())

            # Add formatted text to the plot
            ax.text(0.98, 0.95, formatted_text, transform=ax.transAxes, fontsize=10, color=text_color,
                    verticalalignment='top', horizontalalignment='right', family='monospace',
                    bbox=dict(facecolor=box_color, alpha=0.6, edgecolor='none'))

        # Plotting for latitude and longitude with aligned annotations and lighter blue background
        axs[0].hist(self.lat_lon_counts, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
        axs[0].set_title('Distribution of Configurations (by Unified Risk Index $r$)')
        axs[0].set_xlabel('Criticality Measure $\\Gamma$')
        axs[0].set_ylabel('Number of Occurrences $N$')
        add_aligned_stats_box(axs[0], lat_lon_stats, text_color='black', box_color='aliceblue')

        # Updating TTC near miss plots with aligned annotations and green color scheme
        axs[1].hist(self.ttc_near_miss_counts, bins='auto', color='lightgreen', alpha=0.7, rwidth=0.85)
        axs[1].set_title('Distribution of Configurations (by TTC Near Misses)')
        axs[1].set_xlabel('Criticality Measure $\\Gamma$')
        axs[1].set_ylabel('Number of Occurrences $N$')
        add_aligned_stats_box(axs[1], ttc_stats, text_color='black', box_color='mintcream')

        plt.tight_layout()
        #plt.show()
        plt.savefig('imgs/'+subdir+'.png')
        plt.close()
    

# # Assuming you have adapted your class for CSV if needed.
# base_dir = 'completed_experiments'
# sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# for sub_dir in sub_dirs:
#     file_path = os.path.join(base_dir, sub_dir, 'config.csv')  # or 'config.csv' based on your actual file format
#     analyzer = EdgeCaseAnalyzerFromJSON(file_path)
#     analyzer._plot_distributions(save_path=os.path.join(base_dir, sub_dir, 'distribution.png'))
        
if __name__ == "__main__":
    base_directory = "exp_Files"

    # Check if the base directory exists and iterate through its subdirectories

    for subdir in os.listdir(base_directory):
        analyzer = EdgeCaseAnalyzerFromJSON(os.path.join(base_directory, subdir, 'config.csv'))
        analyzer._plot_distribution_info(base_directory, subdir)
