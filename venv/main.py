import numpy as np
from numpy.linalg import inv
from sympy.physics.quantum import TensorProduct
from sympy import *

def swap_col(matrix, c1, c2):
    a = matrix.copy()
    a[:, c1], a[:, c2] = a[:, c2], a[:, c1].copy()
    return a

def swap_row(a, r1, r2):
    matrix = a.copy()
    matrix[r1, :], matrix[r2, :] = matrix[r2, :], matrix[r1, :].copy()
    return matrix

v = [1, 2, 3, 7, 9, 0, 1, 0, 6]

size_no_zero = np.count_nonzero(v)

matrix = np.reshape(v, (3, 3))

transMatrix = np.transpose(matrix)

reverseMatrix = inv(matrix)
#print('Size of non zero elements: ' + str(size_no_zero))
#print('Matrix from list: \n' + str(matrix))
#print('Transpone matrix: \n' + str(transMatrix))
#print('Reverse matrix: \n' + str(reverseMatrix))
#na trojkatach da smieci bo jest duzo przyblizenie
#print('Check reverse matrix: \n' + str(np.dot(reverseMatrix, matrix)))
#print('Multiply skalar: \n' + str(np.multiply(matrix, 3)))

switchCol = swap_col(matrix, 0, 1)
#print('Switch cols from matrix to: \n ' + str(switchCol))

switchRowMatrix = swap_row(matrix, 0, 1)
#print('Switch rows from matrix to: \n ' + str(switchRowMatrix))

#ILOCZYN TENSOROWY

jednostkowa = Matrix([[1,0], [0, 1]])
i = Matrix([[1,0], [-1, 0]])
id_i = TensorProduct(jednostkowa, i)
#print(id_i)

#Functional operations on matrix
matrix_func = [ 2*x + 1 for x in id_i ]
#print('Func matrix: \n ' + str(np.reshape(matrix_func, (4, 4))))

#TRANSOFRMATA REEDA MULLERA

# n razy iloczyn tensorowy
matrix = np.reshape([[1, 0], [1, 1]], (2, 2))
matrix_3 = TensorProduct(matrix, TensorProduct(matrix, matrix))

pprint('Matrix po 3 tensoarach: \n' + str(matrix_3))

f = Matrix([1, 0,0, 1,1, 0, 1, 1])
print('Func Trans matrix: \n' + str(f))
s_f = [x%2 for x in matrix_3 * f]
pprint(s_f)

#Macierz jednostkowa
id = [int(x) for x in Matrix(np.eye(5))]
pprint(id)


