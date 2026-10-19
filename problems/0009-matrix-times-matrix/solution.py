def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	

    if len(a[0]) != len(b):
        return -1
    
    result = [[] for _ in range(len(a))]
    
    for i, row in enumerate(a):
        for col in zip(*b):
            dot = sum(x*y for x, y in zip(row, col))
            result[i].append(dot)

    return result

