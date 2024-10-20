import numpy as np

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Create a very long vector with very small values and normalize it to have a norm of 1
vector = np.random.rand(100000) * 1e-15
long_vector_float64 = vector.astype(np.float64)
# long_vector_float64 /= np.linalg.norm(long_vector_float64)  # Normalize to have norm 1
long_vector_float128 = vector.astype(np.float128)
# long_vector_float128 /= np.linalg.norm(long_vector_float128)
print(f"Long vector (float64): {long_vector_float64[:10]}")  # Print the first 10 elements for comparison
print(f"Long vector (float128): {long_vector_float128[:10]}")  # Print the first 10 elements for comparison

matrix_rows = 10000
matrix_cols = 100000
chunk_size = 1000
result_float64 = np.zeros(matrix_rows, dtype=np.float64)
result_float128 = np.zeros(matrix_rows, dtype=np.float128)
for i in range(0, matrix_cols, chunk_size):
    # Select a chunk of the matrix and corresponding vector elements
    matrix_chunk = np.random.rand(matrix_rows, chunk_size)
    matrix_chunk64 = matrix_chunk.astype(np.float64) * 1e-2
    matrix_chunk128 = matrix_chunk.astype(np.float128) * 1e-2
    if i < 10:
        print(f"matrix chunk 64 {matrix_chunk64[0][:10]}")
        print(f"matrix chunk 128 {matrix_chunk128[0][:10]}")
        print(f"matrix chunk {matrix_chunk[0][:10]}")
    vector_chunk64 = long_vector_float64[i:i + chunk_size]
    vector_chunk128 = long_vector_float128[i:i + chunk_size]
    result_float64 += np.dot(matrix_chunk, vector_chunk64)
    result_float128 += np.dot(matrix_chunk, vector_chunk128)

# Now compare the results
print("Result (float64):", result_float64[:10])  # Print the first 10 results for comparison
print("Result (float128):", result_float128[:10])  # Print the first 10 results for comparison

# Check the difference between the two results
difference = np.abs(result_float128 - result_float64)
print("Maximum difference between float64 and float128 results:", np.max(difference))
print("Average difference between float64 and float128 results:", np.mean(difference))
