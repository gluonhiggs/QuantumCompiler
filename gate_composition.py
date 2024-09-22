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

# Define the gate set with simplified names
gate_set = {
    "rxp": rotation_gate('x', np.pi / 128),
    "rxn": rotation_gate('x', -np.pi / 128),
    "ryp": rotation_gate('y', np.pi / 128),
    "ryn": rotation_gate('y', -np.pi / 128),
    "rzp": rotation_gate('z', np.pi / 128),
    "rzn": rotation_gate('z', -np.pi / 128)
}

# Simplified gate sequence (as per your suggestion)
gate_sequence = [
    "rxn", "rxn", "rxn", "rxn", "rxn", "rxn", "rxn", "rxn", "rxn", "rxn",
    "rxn", "rxn", "rzp", "rxn", "rzp", "rxn", "rzp", "rxn", "rzp", "rxn",
    "rzp", "rxn", "rxn", "rzp", "rzp", "rxn", "rzp", "rxn", "rzp", "rxn",
    "rzp", "rzp", "rxn", "rzp", "rxn", "rzp", "rxn", "rzp", "rzp", "rxn",
    "rzp", "rxn", "rzp", "rzp", "rxn", "rzp", "rxn", "rzp", "rzp", "rxn",
    "rzp", "rzp", "rxn", "rzp", "rzp", "rxn", "rzp", "rzp", "rxn", "rzp",
    "rzp", "rxn", "rzp", "rzp", "rzp", "rxn", "rzp", "rzp", "rxn", "rzp",
    "rzp", "rzp", "rxn"
]

# Initialize the result as an identity matrix
result_matrix = np.eye(2, dtype=complex)

# Multiply the gates in sequence
for gate_name in gate_sequence:
    result_matrix = np.dot(result_matrix, gate_set[gate_name])

# Print the final resultant matrix
print("Resultant matrix after applying the simplified gate sequence:")
print(result_matrix)
