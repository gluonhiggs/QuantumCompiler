import numpy as np
def get_haar_random_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

# Get the Haar random unitary matrix
U = get_haar_random_unitary()
print(U)

# Check if U is unitary
is_unitary = np.allclose(U @ U.conj().T, np.eye(2))
print(U @ U.conj().T)
print(is_unitary)
