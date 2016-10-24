# -*- coding: utf-8 -*-

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2], [3, 4]])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[7, 8, 9], [10, 11, 12]])
h = np.array([[5, 6], [7, 8]])

print(a)
print(b)

e = np.hstack((c, d))  # or e = np.c_[c,d]
f = np.vstack((c, d))  # or f= np.r_[c,d]
g = np.transpose(c)  # or a.transpose()

print('''Stacking horizontally:
{}'''.format(e))
print('''Stacking vertically:
{}'''.format(f))
print('''Transposition:
{}'''.format(g))

f=b.dot(h)  # matrices product in an efficient way

print('''Matrices product:
{}'''.format(f))

g = c*d  # element wise product
print('''Element wise product:
{}'''.format(g))

