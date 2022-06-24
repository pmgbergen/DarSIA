import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


img = cv2.imread("images/originals/Profilbilde.jpg", 0)
# img = cv2.resize(img, (0,0), fx = 0.2, fy = 0.2)

ny = img.shape[0]
nx = img.shape[1]

leny = 1.0
lenx = nx / ny
h = leny / ny
ell = 0.0000001

# def indexToNb(i,j):
#     return i + ny*j

# def nbToIndex(k):
#     i = k%ny
#     j = int((k-i)/ny)
#     return i, j


numNodes = int(ny * nx)

# bdryNodes1 = np.arange(ny)
# bdryNodes2 = np.arange(ny,numNodes-ny,ny)
# bdryNodes3 = np.arange(ny-1,numNodes,ny)
# bdryNodes4 = np.arange(numNodes-ny,numNodes)
# bdryNodes = np.concatenate((bdryNodes1, bdryNodes2, bdryNodes3, bdryNodes4), axis=None)

rowsDiag = np.arange(numNodes)
rowsOffDiag1 = np.arange(numNodes - 1)
rowsOffDiag2 = np.arange(1, numNodes)
rowsOffDiag3 = np.arange(numNodes - ny)
rowsOffDiag4 = np.arange(ny, numNodes)
rows = np.concatenate(
    (rowsDiag, rowsOffDiag1, rowsOffDiag2, rowsOffDiag3, rowsOffDiag4), axis=None
)

colsDiag = np.arange(numNodes)
colsOffDiag1 = np.arange(1, numNodes)
colsOffDiag2 = np.arange(numNodes - 1)
colsOffDiag3 = np.arange(ny, numNodes)
colsOffDiag4 = np.arange(numNodes - ny)
cols = np.concatenate(
    (colsDiag, colsOffDiag1, colsOffDiag2, colsOffDiag3, colsOffDiag4), axis=None
)

entriesDiag = 1.0 + 4 * ell / (h**2) * np.ones(numNodes)
entriesOffDiag = -ell / h**2 * np.ones(len(rows) - numNodes)
entries = np.concatenate((entriesDiag, entriesOffDiag), axis=None)

matrix = coo_matrix((entries, (rows, cols))).tocsr()
# matrix[bdryNodes,:] = 0.0
# matrix[bdryNodes,bdryNodes] = 1
# matrix = matrix.tocsr()
rhs = np.reshape(img, numNodes, order="F")

x = spsolve(matrix, rhs)
img2 = np.reshape(x, (ny, nx), order="F")
print(img2.shape)

print(x)
print(rhs)
cv2.imwrite("images/modified/Profilbilde2.jpg", img2)
