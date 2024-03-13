import numpy as np

# Given probability array
prob_array = np.array([2.32061126e-21, 2.81466355e-01, 3.12534810e-13, 2.32061126e-21,
                        2.32061126e-21, 2.32061126e-21, 7.30527984e-06, 2.32061126e-21,
                        2.32061126e-21, 2.32061126e-21, 5.27317419e-20, 2.32061126e-21,
                        2.32061126e-21, 9.26290831e-19, 7.43802927e-10, 2.32061126e-21,
                        2.32061126e-21, 9.51302521e-10, 2.32061126e-21, 4.36395926e-01,
                        2.32061126e-21, 1.28689083e-08, 2.32061126e-21, 2.32061126e-21,
                        2.32061126e-21, 6.59065437e-10, 2.32061126e-21, 2.32061126e-21,
                        2.32061126e-21, 2.32061126e-21, 2.43664182e-21, 2.32061126e-21,
                        2.32061126e-21, 1.09519422e-05, 2.32061126e-21, 2.32061126e-21,
                        2.32061126e-21, 2.32061126e-21, 2.82119447e-01, 2.32061126e-21,
                        2.43664182e-21, 5.20235087e-20, 2.32061126e-21, 2.32061126e-21,
                        4.09280298e-17, 2.32061126e-21, 3.12535549e-13, 2.32061126e-21,
                        2.32061126e-21, 2.32061126e-21])

max_prob = np.max(prob_array)
min_prob = np.min(prob_array[prob_array > 0])

# Ensure max and min probabilities are within an order of magnitude
if max_prob > 10 * min_prob:
    # Adjust probabilities to bring them within an order of magnitude
    prob_array = np.clip(prob_array, max_prob / 10, max_prob)

# Normalize to sum to 1
prob_array /= prob_array.sum()
print(prob_array)
