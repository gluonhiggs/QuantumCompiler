import numpy as np

# Define rotation gates
def rotation_gate(axis, angle):
    if axis == 'x':
        return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                         [-1j * np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'y':
        return np.array([[np.cos(angle / 2), -np.sin(angle / 2)],
                         [np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'z':
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0, np.exp(1j * angle / 2)]], dtype=complex)

# Gate set
gate_set = [
    rotation_gate('x', np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y', np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z', np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]

# Function to compute Hilbert-Schmidt inner product
def hilbert_schmidt_inner_product(A, B):
    return np.trace(np.dot(A.conj().T, B))

# Compute inner products for all pairs of gates
n = len(gate_set)
inner_product_matrix = np.zeros((n, n), dtype=complex)

for i in range(n):
    for j in range(n):
        inner_product_matrix[i, j] = hilbert_schmidt_inner_product(gate_set[i], gate_set[j])
# Format options for better printing
np.set_printoptions(precision=17, suppress=True)
# Display the Hilbert-Schmidt inner product matrix
print("Hilbert-Schmidt Inner Product Matrix:")
print(inner_product_matrix)
