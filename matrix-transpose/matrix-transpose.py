import numpy as np

def matrix_transpose(A):
    if len(A) == 0:
        return np.array([])

    n = len(A)        # rows
    m = len(A[0])     # columns

    A_T = [[0]*n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            A_T[j][i] = A[i][j]

    return np.array(A_T)