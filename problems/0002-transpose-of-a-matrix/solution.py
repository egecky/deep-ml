def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    dim1 = len(a)
    dim2 = len(a[0])

    transposed = [[] for _ in range(dim2)]

    for i in range(dim1):
        for j in range(dim2):
            transposed[j].append(a[i][j])

    return transposed

