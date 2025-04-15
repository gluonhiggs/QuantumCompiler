import numpy as np
from scipy.linalg import expm
import heapq

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
    def __init__(self, input_dim=128, output_dim=1):
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
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)

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

            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            next_q = target_dqn(next_states).max(1)[0] - 1
        loss = ((q_values - (next_q + values))**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 0.02:
            break

    target_dqn.load_state_dict(dqn.state_dict())

def aq_star_search(target_unitary, dqn, action_space, max_depth=500):
    # Use a unique ID for each entry to handle potential duplicate states/f-scores
    # And store (-f_score, g, id, unitary, gates) to use heapq (min-heap)
    # We negate f_score because heapq is a min-heap, and we want to maximize f
    import heapq
    import itertools
    counter = itertools.count() # Unique sequence numbers

    # Initial state: Start from target, g=0 (0 steps from target)
    initial_state_flat = np.concatenate([target_unitary.real.flatten(), target_unitary.imag.flatten()])
    initial_state_tensor = torch.FloatTensor(initial_state_flat).unsqueeze(0)
    with torch.no_grad():
        # *** CRITICAL: Check if DQN output should be single value (V) or max(Q) ***
        # Assuming DQN outputs Q-values for now, as per your code structure
        # h_score = dqn(initial_state_tensor).max().item()
        # If DQN learns V(s) directly (output_dim=1), use:
        # h_score = dqn(initial_state_tensor).item()
        # Let's proceed assuming Q-values for now based on your code structure
        # but highlight this potential mismatch with the paper later.
        h_score = dqn(initial_state_tensor).max().item()

    initial_g = 0
    initial_f = initial_g + h_score # f = g + h. Note g=0 initially.
    unique_id = next(counter)

    # Heap stores: (-f_score, g, unique_id, unitary, gates_list)
    # Negate f_score for max-heap behavior using min-heap
    open_set_heap = [(-initial_f, initial_g, unique_id, target_unitary, [])]
    heapq.heapify(open_set_heap)

    # Keep track of visited states (optional but good practice for A*)
    # Key: hashable representation of unitary (e.g., tobytes), Value: best g-score
    visited = {}
    visited[target_unitary.tobytes()] = initial_g

    best_unitary_at_identity = None # Store the state closest to Identity found so far
    best_gates_to_identity = None
    min_dist_to_identity = 1.0 # Max possible distance is 1-fidelity = 1

    print(f"Starting AQ* search from target...")

    for i in range(max_depth):
        if not open_set_heap:
            print("Search space exhausted.")
            break

        # Get node with highest f-score (lowest -f_score from min-heap)
        neg_f, g, _, current_unitary, gates = heapq.heappop(open_set_heap)
        #f = -neg_f

        # Check if goal reached (current_unitary is close to Identity)
        fidelity_to_identity = abs(np.trace(np.eye(8) @ np.conj(current_unitary).T)) / 8
        dist_to_identity = 1.0 - fidelity_to_identity
        #print(f"Step {i}, Popped node. Fidelity to I: {fidelity_to_identity:.6f}, g={g}, f={f:.4f}, Heap size: {len(open_set_heap)}")


        if fidelity_to_identity >= 1 - 1e-4:
            print(f"Solution found at step {i} with fidelity: {fidelity_to_identity:.6f}")
            # Reverse gates because search went Target -> Identity
            return gates[::-1]

        # Track the best state encountered in terms of closeness to Identity
        if dist_to_identity < min_dist_to_identity:
             min_dist_to_identity = dist_to_identity
             best_unitary_at_identity = current_unitary
             best_gates_to_identity = gates
             print(f"  New best state found: Dist to I: {dist_to_identity:.6f}, Path length: {-g}")


        # Expand node: Apply actions (inverse gates conceptually)
        # In the paper, they apply g^-1. If action_space contains inverses, applying 'a' might be equivalent.
        # Let's stick to a @ current_unitary as in your code for now.
        for a_idx, a in enumerate(action_space):
            # Apply the gate 'a'
            # Note: Paper applies g^-1. If 'a' is g, we should use a.conj().T
            # Let's assume a @ current_unitary is the intended operation for now.
            new_unitary = a @ current_unitary
            new_unitary_bytes = new_unitary.tobytes()

            # g score increases by 1 step away from target, which is -1 in cost terms
            new_g = g - 1

            # Check if visited with a better or equal g-score
            if new_unitary_bytes in visited and visited[new_unitary_bytes] <= new_g:
                continue # Already found a shorter or equal path to this state

            # Calculate heuristic h(new_unitary) using DQN
            new_state_flat = np.concatenate([new_unitary.real.flatten(), new_unitary.imag.flatten()])
            new_state_tensor = torch.FloatTensor(new_state_flat).unsqueeze(0)
            with torch.no_grad():
                # *** Potential Mismatch Point ***
                # Using max(Q(s',a')) as h(s')
                h_score = dqn(new_state_tensor).max().item()
                # If DQN learns V(s') directly (output_dim=1), use:
                # h_score = dqn(new_state_tensor).item()

            new_f = new_g + h_score

            # Add to visited list and priority queue
            visited[new_unitary_bytes] = new_g
            new_gates = gates + [a_idx]
            unique_id = next(counter)
            heapq.heappush(open_set_heap, (-new_f, new_g, unique_id, new_unitary, new_gates))

    print(f"Max depth {max_depth} reached.")
    print(f"Best state found had distance to Identity: {min_dist_to_identity:.6f}")
    # Return best path found if exact solution wasn't reached
    return best_gates_to_identity[::-1] if best_gates_to_identity is not None else None
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
