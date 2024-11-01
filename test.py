# "target_U": "[0.417127465+0.169897933j, -0.778189692+0.437675877j][-0.098094679+0.887421435j, 0.319635975+0.317322349j]", 
# "approximate_U": "[-0.372576157+0.062910539j, 0.339503182-0.861374983j][-0.339503182-0.861374983j, -0.372576157-0.062910539j]"}
import numpy as np
target_U = np.array([[0.417127465+0.169897933j, -0.778189692+0.437675877j],
    [-0.098094679+0.887421435j, 0.319635975+0.317322349j]], dtype=complex)
approximate_U = np.array([[-0.372576157+0.062910539j, 0.339503182-0.861374983j],
    [-0.339503182-0.861374983j, -0.372576157-0.062910539j]], dtype=complex)

def average_gate_fidelity(U, V):
    d = U.shape[0]
    U_dagger = np.conjugate(U.T)
    trace_U_dagger_V = np.trace(np.dot(U_dagger, V))
    fidelity = (np.abs(trace_U_dagger_V)**2 + d) / (d * (d + 1))
    return fidelity

fidelity = average_gate_fidelity(approximate_U, target_U)

print(fidelity)