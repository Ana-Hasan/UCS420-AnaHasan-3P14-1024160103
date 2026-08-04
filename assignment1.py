import numpy as np
from itertools import product
from PIL import Image

print("----- (a) Reverse Array -----")
arr = np.array([1, 2, 3, 6, 4, 5])
print(arr[::-1])

print("\n----- (b) Flatten Array -----")
array1 = np.array([[1, 2, 3],
                   [2, 4, 5],
                   [1, 2, 3]])

print(array1.flatten())
print(array1.ravel())

print("\n----- (c) Compare Arrays -----")
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[1, 2],
                 [3, 4]])

print(np.array_equal(arr1, arr2))
print(arr1 == arr2)

print("\n----- (d)(i) Most Frequent Value in x -----")
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])

values, counts = np.unique(x, return_counts=True)
most = values[np.argmax(counts)]
indices = np.where(x == most)

print("Most Frequent Value:", most)
print("Indices:", indices)

print("\n----- (d)(ii) Most Frequent Value in y -----")
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])

values, counts = np.unique(y, return_counts=True)
most = values[np.argmax(counts)]
indices = np.where(y == most)

print("Most Frequent Value:", most)
print("Indices:", indices)

print("\n----- (e) Matrix Operations -----")
gfg = np.matrix('[4,1,9;12,3,1;4,5,6]')

print("Sum of all elements:")
print(np.sum(gfg))

print("Row-wise sum:")
print(np.sum(gfg, axis=1))

print("Column-wise sum:")
print(np.sum(gfg, axis=0))

print("\n----- (f) Matrix Calculations -----")
n_array = np.array([[55, 25, 15],
                    [30, 44, 2],
                    [11, 45, 77]])

print("Sum of diagonal elements:")
print(np.trace(n_array))

print("Eigenvalues:")
print(np.linalg.eigvals(n_array))

eigenvalues, eigenvectors = np.linalg.eig(n_array)

print("Eigenvectors:")
print(eigenvectors)

print("Inverse:")
print(np.linalg.inv(n_array))

print("Determinant:")
print(np.linalg.det(n_array))

print("\n----- (g)(i) Matrix Multiplication and Covariance -----")
p = np.array([[1, 2],
              [2, 3]])

q = np.array([[4, 5],
              [6, 7]])

print("Matrix Multiplication:")
print(np.matmul(p, q))

print("Covariance:")
print(np.cov(p, q))

print("\n----- (g)(ii) Matrix Multiplication and Covariance -----")
p = np.array([[1, 2],
              [2, 3],
              [4, 5]])

q = np.array([[4, 5, 1],
              [6, 7, 2]])

print("Matrix Multiplication:")
print(np.matmul(p, q))

print("Covariance:")
print(np.cov(p.T, q))

print("\n----- (h) Inner, Outer and Cartesian Product -----")
x = np.array([[2, 3, 4],
              [3, 2, 9]])

y = np.array([[1, 5, 0],
              [5, 10, 3]])

print("Inner Product:")
print(np.inner(x, y))

print("Outer Product:")
print(np.outer(x, y))

cartesian = np.array(list(product(x.flatten(), y.flatten())))

print("Cartesian Product:")
print(cartesian)


import numpy as np

print("----- (a) Array Operations -----")

array = np.array([[1, -2, 3],
                  [-4, 5, -6]])

print("Absolute Value:")
print(np.abs(array))

print("\n25th, 50th and 75th Percentile (Flattened):")
print(np.percentile(array, [25, 50, 75]))

print("\n25th, 50th and 75th Percentile (Column-wise):")
print(np.percentile(array, [25, 50, 75], axis=0))

print("\n25th, 50th and 75th Percentile (Row-wise):")
print(np.percentile(array, [25, 50, 75], axis=1))

print("\nMean (Flattened):")
print(np.mean(array))

print("Mean (Column-wise):")
print(np.mean(array, axis=0))

print("Mean (Row-wise):")
print(np.mean(array, axis=1))

print("\nMedian (Flattened):")
print(np.median(array))

print("Median (Column-wise):")
print(np.median(array, axis=0))

print("Median (Row-wise):")
print(np.median(array, axis=1))

print("\nStandard Deviation (Flattened):")
print(np.std(array))

print("Standard Deviation (Column-wise):")
print(np.std(array, axis=0))

print("Standard Deviation (Row-wise):")
print(np.std(array, axis=1))

print("\n----- (b) Floor, Ceiling, Truncate and Round -----")

a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])

print("Floor:")
print(np.floor(a))

print("Ceiling:")
print(np.ceil(a))

print("Truncated:")
print(np.trunc(a))

print("Rounded:")
print(np.round(a))


import numpy as np

print("----- (a) Array Operations -----")

array = np.array([10, 52, 62, 16, 16, 54, 453])

print("Sorted Array:")
print(np.sort(array))

print("\nIndices of Sorted Array:")
print(np.argsort(array))

print("\n4 Smallest Elements:")
print(np.sort(array)[:4])

print("\n5 Largest Elements:")
print(np.sort(array)[-5:])

print("\n----- (b) Integer and Float Elements -----")

array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

print("Integer Elements Only:")
print(array[array == array.astype(int)])

print("\nFloat Elements Only:")
print(array[array != array.astype(int)])


def img_to_array(path):
    img = Image.open(path)

    arr = np.array(img)

    if len(arr.shape) == 3:
        print("RGB Image")
        np.savetxt("rgb_image.txt", arr.reshape(-1, 3), fmt="%d")
        print("Saved as rgb_image.txt")

    elif len(arr.shape) == 2:
        print("Grayscale Image")
        np.savetxt("grayscale_image.txt", arr, fmt="%d")
        print("Saved as grayscale_image.txt")

    else:
        print("Unsupported image format")

        