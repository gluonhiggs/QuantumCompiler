import numpy as np

def rotation_gate(axis, angle):
    if axis == 'x':
        return np.array([
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            [-1j * np.sin(angle / 2), np.cos(angle / 2)]
        ], dtype=complex)
    elif axis == 'y':
        return np.array([
            [np.cos(angle / 2), -np.sin(angle / 2)],
            [np.sin(angle / 2), np.cos(angle / 2)]
        ], dtype=complex)
    elif axis == 'z':
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)

# Gate descriptions and matrices
gate_descriptions = [
    "rxp",
    "rxn",
    "ryp",
    "ryn",
    "rzp",
    "rzn"
]

gate_matrices = [
    rotation_gate('x', np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y', np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z', np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]

gate_dict = dict(zip(gate_descriptions, gate_matrices))

# Gate sequence provided by you
# Gate sequence provided by you
gate_sequence_descriptions = ['rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn', 'rzn']


# Map gate descriptions to matrices
gate_sequence_matrices = [gate_dict[desc] for desc in gate_sequence_descriptions]

# Compute U_S
U_S = np.eye(2, dtype=complex)
for gate in gate_sequence_matrices:
    U_S = np.dot(U_S, gate)

print("Resultant matrix after applying the gate sequence:")
print(U_S)
# Get target unitary U
def get_fixed_target_unitary():
    return np.array([
        [0.76749896 - 0.43959894j, -0.09607122 + 0.45658344j],
        [0.09607122 + 0.45658344j,  0.76749896 + 0.43959894j]
    ], dtype=complex)

U = get_fixed_target_unitary()

# Compute operator norm error
def operator_norm_error(U_S, U):
    difference = U_S - U
    singular_values = np.linalg.svd(difference, compute_uv=False)
    op_norm_error = np.max(singular_values)
    return op_norm_error

error = operator_norm_error(U_S, U)
print(f"Operator norm error between U_S and U: {error}")
