import numpy as np
V1 = (1/np.sqrt(5)) * np.array([[1, 2j],
                                    [2j, 1]], dtype=complex)

observation = np.concatenate([V1.real.flatten(), V1.imag.flatten()])
print(observation)
U_n_real = observation[:4]
U_n_imag = observation[4:]
U_n = U_n_real + 1j * U_n_imag
U_n = U_n.reshape(2, 2)
print(U_n)