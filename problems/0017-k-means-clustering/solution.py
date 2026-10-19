def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	
	def euclidean_distance(point1: tuple[float, ...], point2: tuple[float, ...]) -> float:
		return sum((x-y)**2 for x, y in zip(point1, point2))
	
	final_centroids = initial_centroids.copy()

	# iteration
	for _ in range(max_iterations):
		assignments = [0 for _ in points]

		# assignment step
		for i, point in enumerate(points):
			minval = euclidean_distance(point, final_centroids[0])
			for j, centr in enumerate(final_centroids[1:]):
				dist = euclidean_distance(point, centr)
				if dist < minval:
					assignments[i] = j+1
					minval = dist
		
		# update step
		accumulator = [[] for _ in range(len(final_centroids))]
		for i, assignment in enumerate(assignments):
			accumulator[assignment].append(points[i])
		
		for i, coll in enumerate(accumulator):
			if coll:
				final_centroids[i] = tuple([sum(p)/ len(coll) for p in zip(*coll)])
			
	return final_centroids