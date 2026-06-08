import numpy as np

arr = np.load("Main Pipeline Final/Data Files/seed_qpsk.npy")

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("First 20 elements:")
print(arr)