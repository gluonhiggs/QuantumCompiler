import numpy as np

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]])  # Identity matrix
X = np.array([[0, 1], [1, 0]])  # Pauli-X (bit-flip)
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z

# Define single-qubit rotation matrices
def R_x(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X

def R_y(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Y

# Define XX gate for two qubits
def XX_gate(theta):
    XX = np.kron(X, X)  # Tensor product of Pauli-X matrices for two qubits
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * XX

# Define the RZ rotation matrix
def R_z(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

# Use the decomposition given for RZ
def RZ_decomposed(theta, v=1):
    # Decompose RZ(theta) into RY(-v*pi/2) -> RX(v*theta) -> RY(v*pi/2)
    RY1 = R_y(-v * np.pi / 2)  # RY(-v*pi/2)
    RX = R_x(v * theta)        # RX(v*theta)
    RY2 = R_y(v * np.pi / 2)   # RY(v*pi/2)
    
    return RY1 @ RX @ RY2

# Test the decomposition by comparing with the actual RZ gate
theta_test = np.pi / 3  # Example angle for RZ
RZ_actual = R_z(theta_test)
print(f"R_z matrix for theta = {theta_test}:\n{RZ_actual}")

RZ_decomposed_matrix = RZ_decomposed(theta_test)

# Check if the decomposed RZ matches the actual RZ matrix
rz_result = np.allclose(RZ_actual, RZ_decomposed_matrix)

print(f"Decomposed RZ matches actual RZ: {rz_result}")
print("Decomposed RZ matrix:")
print(RZ_decomposed_matrix)