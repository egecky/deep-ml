import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	Tarr = np.array(T)
	Aarr = np.array(A)
	Sarr = np.array(S)

	if np.linalg.matrix_rank(Tarr) != len(Tarr[0]) or np.linalg.matrix_rank(Sarr) != len(Sarr[0]):
		return -1
	
	return (np.linalg.inv(Tarr) @ Aarr @ Sarr).tolist()