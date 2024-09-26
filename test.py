import numpy as np

# Generate a random Haar-distributed pure state
def random_pure_state(dim=2):
    """Generate a random pure state |psi> in a Hilbert space of given dimension."""
    # Create a random complex vector
    state = (np.random.randn(dim) + 1j * np.random.randn(dim))
    # Normalize the vector to get a pure state
    return state / np.linalg.norm(state)

# Compute the fidelity between two states
def fidelity(psi1, psi2):
    """Fidelity between two pure states |psi1> and |psi2>."""
    return np.abs(np.dot(np.conjugate(psi1), psi2))**2

# Compute the average gate fidelity via Monte Carlo approximation
def average_gate_fidelity(U, V, num_samples=100000):
    """Approximate the average gate fidelity between unitary matrices U and V."""
    dim = U.shape[0]
    fidelities = []

    for _ in range(num_samples):
        # Generate a random pure state |psi>
        psi = random_pure_state(dim)
        print(f'random pure state: {psi}')
        
        # Apply the unitaries to the state
        U_psi = np.dot(U, psi)
        V_psi = np.dot(V, psi)
        
        # Compute fidelity between U|psi> and V|psi>
        fidelities.append(fidelity(U_psi, V_psi))
    
    # Return the average fidelity
    return np.mean(fidelities)


U = np.eye(2) 
V = np.array([[0.99, -0.1], [0.1, 0.99]])  

# Approximate the average gate fidelity
avg_fidelity = average_gate_fidelity(U, V, num_samples=100000)
print(f"Average Gate Fidelity: {avg_fidelity:.6f}")
