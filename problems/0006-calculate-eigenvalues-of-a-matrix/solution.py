from math import sqrt

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	b = matrix[0][0] + matrix[1][1]

	disc = sqrt(b**2 - 4*c)
	return [(b + disc)/2, (b - disc)/2]