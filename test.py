import numpy as np
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

gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x', np.pi / 128),    # rxp
    rotation_gate('x', -np.pi / 128),   # rxn
    rotation_gate('y', np.pi / 128),    # ryp
    rotation_gate('y', -np.pi / 128),   # ryn
    rotation_gate('z', np.pi / 128),    # rzp
    rotation_gate('z', -np.pi / 128)    # rzn
]

sequence = ["ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "ryn", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "rzp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "ryp", "rxn", "ryp", "rzn", "ryp", "rzn", "ryp", "rzn", "ryp", "rzn", "ryp", "rxn", "rzn", "rzn", "ryp", "rzn", "ryp", "rzn", "ryp", "rzn", "rxn", "rzn", "rzn", "ryp", "rzn", "rzn", "ryp", "rxn", "rzn", "rzn", "rzn", "rzn", "ryp", "rzn", "rxn", "rzn", "rzn", "rzn", "rzn", "rxn", "rzn", "rzn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rzn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rxn", "rzn", "rzn", "rxn", "rzn", "rxn", "rzn", "rxn", "rzn", "rxn", "rzn", "rzn"]

# Calculate the product of the matrices in the sequence
result = np.eye(2)
for gate in sequence:
    print(gate_descriptions.index(gate))
    print(gate_matrices[gate_descriptions.index(gate)])
    result = np.dot(gate_matrices[gate_descriptions.index(gate)], result)

# Print the result
print(result)
