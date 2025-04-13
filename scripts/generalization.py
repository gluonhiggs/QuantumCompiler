import numpy as np
from scipy.linalg import expm

class QuantumCompileEnv:
    def __init__(self, target_unitary, action_space):
        self.target = target_unitary  # e.g., QFT unitary (8x8)
        self.action_space = action_space  # List of gates in \tilde{\mathcal{G}}
        self.current_unitary = np.eye(8, dtype=complex)  # Start with I_8
        self.max_steps = 500
        self.step_count = 0

    def reset(self):
        self.current_unitary = np.eye(8, dtype=complex)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # Flatten unitary into real and imaginary parts
        return np.concatenate([self.current_unitary.real.flatten(), self.current_unitary.imag.flatten()])

    def step(self, action_idx):
        # Apply gate corresponding to action_idx
        gate = self.action_space[action_idx]
        self.current_unitary = gate @ self.current_unitary
        self.step_count += 1

        # Compute reward
        fidelity = abs(np.trace(np.conj(self.target).T @ self.current_unitary)) / 8
        reward = fidelity

        # Done condition
        done = fidelity >= 1 - 1e-4 or self.step_count >= self.max_steps
        return self._get_state(), reward, done, {}

# Define native gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
T = np.diag([1, np.exp(1j * np.pi / 4)])
CZ = np.diag([1, 1, 1, -1]).reshape(4, 4)

# Define action space \tilde{\mathcal{G}}
def RX(theta):
    return expm(-1j * theta / 2 * X)

def RY(theta):
    return expm(-1j * theta / 2 * Y)

action_space = [
    np.kron(np.kron(RX(np.pi/2), I), I),  # RX(pi/2) on Q1
    np.kron(np.kron(I, RX(np.pi/2)), I),  # RX(pi/2) on Q2
    np.kron(np.kron(I, I), RX(np.pi/2)),  # RX(pi/2) on Q3
    # RX(-pi/2) on each qubit
    np.kron(np.kron(RX(-np.pi/2), I), I),  # Q1
    np.kron(np.kron(I, RX(-np.pi/2)), I),  # Q2
    np.kron(np.kron(I, I), RX(-np.pi/2)),  # Q3
    # RY(pi/2) on each qubit
    np.kron(np.kron(RY(np.pi/2), I), I),  # Q1
    np.kron(np.kron(I, RY(np.pi/2)), I),  # Q2
    np.kron(np.kron(I, I), RY(np.pi/2)),  # Q3
    # RY(-pi/2) on each qubit
    np.kron(np.kron(RY(-np.pi/2), I), I),  # Q1
    np.kron(np.kron(I, RY(-np.pi/2)), I),  # Q2
    np.kron(np.kron(I, I), RY(-np.pi/2)),  # Q3
    # T on each qubit
    np.kron(np.kron(T, I), I),  # Q1
    np.kron(np.kron(I, T), I),  # Q2
    np.kron(np.kron(I, I), T),  # Q3
    # T^\dagger on each qubit
    np.kron(np.kron(T.conj().T, I), I),  # Q1
    np.kron(np.kron(I, T.conj().T), I),  # Q2
    np.kron(np.kron(I, I), T.conj().T),  # Q3
    np.kron(CZ, I),  # CZ between Q2-Q3
    np.kron(I, CZ),  # CZ between Q3-Q6
    # Total 20 actions
]

# Target: Three-qubit QFT unitary
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
qft_circuit = QFT(num_qubits=3, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False)

# Convert the circuit to its unitary matrix
target_unitary = Operator(qft_circuit).data

env = QuantumCompileEnv(target_unitary, action_space)
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim=128, output_dim=20):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 8000)
        self.hidden1 = nn.Linear(8000, 3000)
        # 6 residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(3000) for _ in range(6)])
        self.output_layer = nn.Linear(3000, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(8000)
        self.batch_norm2 = nn.BatchNorm1d(3000)

    def forward(self, x):
        x = self.leaky_relu(self.batch_norm1(self.input_layer(x)))
        x = self.leaky_relu(self.batch_norm2(self.hidden1(x)))
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        x = self.leaky_relu(self.batch_norm(self.fc1(x)))
        x = self.batch_norm(self.fc2(x))
        x += residual
        return self.leaky_relu(x)

# Training data generation
def generate_training_data(action_space, L=44, l_star=3):
    W = [np.eye(8, dtype=complex)]  # Start with I
    V = [0]
    data = []

    for l in range(1, L+1):
        W_next = []
        V_next = []
        for w in W:
            for a in action_space:
                w_new = a @ w
                # Compute value: min gates to revert to I (simplified)
                fidelity = abs(np.trace(np.eye(8) @ np.conj(w_new).T)) / 8
                value = -l if fidelity < 1 - 1e-4 else 0
                W_next.append(w_new)
                V_next.append(value)

        # Random sampling for l > l_star
        if l > l_star:
            indices = np.random.choice(len(W_next), size=min(10000, len(W_next)), replace=False)
            W_next = [W_next[i] for i in indices]
            V_next = [V_next[i] for i in indices]

        # Prepare data for DQN: (state, action, value)
        for w, v in zip(W_next, V_next):
            state = np.concatenate([w.real.flatten(), w.imag.flatten()])
            for a_idx, a in enumerate(action_space):
                data.append((state, a_idx, v))

        W = W_next
        V = V_next

    return data

# Training loop
dqn = DQN()
target_dqn = DQN()
target_dqn.load_state_dict(dqn.state_dict())
optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
data = generate_training_data(action_space)

for l in range(1, 45):
    for epoch in range(100 * l):
        batch = np.random.choice(len(data), size=64)
        states, actions, values = zip(*[data[i] for i in batch])
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        values = torch.FloatTensor(values)

        q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_states = []
            for i, a in enumerate(actions):
                # Reconstruct the unitary from the state vector
                state = data[i][0]  # Shape (128,)
                real_part = state[:64].reshape(8, 8)  # First 64 elements -> 8x8 real part
                imag_part = state[64:].reshape(8, 8)  # Next 64 elements -> 8x8 imaginary part
                unitary = real_part + 1j * imag_part  # Reconstruct the 8x8 complex unitary

                # Apply the action (gate) to the unitary
                new_unitary = action_space[a] @ unitary

                # Convert the new unitary back to a state vector
                new_state = np.concatenate([new_unitary.real.flatten(), new_unitary.imag.flatten()])
                next_states.append(new_state)

            next_states = torch.FloatTensor(np.array(next_states))
            next_q = target_dqn(next_states).max(1)[0] - 1
        loss = ((q_values - (next_q + values))**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 0.02:
            break

    target_dqn.load_state_dict(dqn.state_dict())

def aq_star_search(target_unitary, dqn, action_space, max_depth=500):
    open_set = [(target_unitary, [], 0)]  # (unitary, gate sequence, g-score)
    best_unitary = target_unitary
    best_gates = []
    best_fidelity = 0

    for _ in range(max_depth):
        if not open_set:
            break

        # Select node with best f-score
        current, gates, g = max(open_set, key=lambda x: x[2] + dqn(torch.FloatTensor(
                np.concatenate([x[0].real.flatten(), x[0].imag.flatten()])
            ).unsqueeze(0)).max().item())
        state = torch.FloatTensor(np.concatenate([current.real.flatten(), current.imag.flatten()]))
        print("Input shape to DQN:", state.shape)
        open_set.remove((current, gates, g))

        # Check if goal reached
        fidelity = abs(np.trace(np.eye(8) @ np.conj(current).T)) / 8
        if fidelity >= 1 - 1e-4:
            print(f"Solution found with fidelity: {fidelity}")
            return gates[::-1]  # Reverse to get forward sequence

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_unitary = current
            best_gates = gates

        # Expand node
        for a_idx, a in enumerate(action_space):
            new_unitary = a @ current
            new_g = g - 1
            state = torch.FloatTensor(np.concatenate([new_unitary.real.flatten(), new_unitary.imag.flatten()])).unsqueeze(0)
            f_score = new_g + dqn(state).max().item()
            new_gates = gates + [a_idx]
            open_set.append((new_unitary, new_gates, f_score))

    return best_gates[::-1] if best_gates else None  # Return best found if max depth reached

# Run AQ* search
dqn.eval()
gate_sequence = aq_star_search(target_unitary, dqn, action_space)

# Convert gate sequence to circuit
# Convert gate sequence to circuit
if gate_sequence is not None:
    circuit = []
    cz_count = 0
    for a_idx in gate_sequence:
        gate = action_space[a_idx]
        if a_idx in [18, 19]:  # CZ gates are at indices 18 (Q2-Q3) and 19 (Q3-Q6)
            cz_count += 1
        circuit.append(gate)
        print(f"Compiled circuit with {cz_count} CZ gates")
else:
    print("No solution found")
