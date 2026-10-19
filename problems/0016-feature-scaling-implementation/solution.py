import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	
	standardized_data = (data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True)

	normalized_data = (data - np.min(data, axis=0, keepdims=True)) / (np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))


	return standardized_data, normalized_data