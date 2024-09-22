# Result

## Fixed target unitary

### Improvement (v2)
```bash
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.average_gate_fidelity to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.average_gate_fidelity` for environment variables or `env.get_wrapper_attr('average_gate_fidelity')` that will search the reminding wrappers.
  logger.warn(
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.U_n to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.U_n` for environment variables or `env.get_wrapper_attr('U_n')` that will search the reminding wrappers.
  logger.warn(
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.target_U to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.target_U` for environment variables or `env.get_wrapper_attr('target_U')` that will search the reminding wrappers.
  logger.warn(
Final Fidelity: 0.991499662833286
Sequence Length: 70
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.tolerance to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.tolerance` for environment variables or `env.get_wrapper_attr('tolerance')` that will search the reminding wrappers.
  logger.warn(
Successfully approximated the target unitary.
Gate Sequence:
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp']
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.gate_set to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.gate_set` for environment variables or `env.get_wrapper_attr('gate_set')` that will search the reminding wrappers.
  logger.warn(
Resultant matrix after applying the gate sequence:
[[ 0.8217011-0.37708926j -0.0987385+0.41576641j]
 [ 0.0987385+0.41576641j  0.8217011+0.37708926j]]
Success rate: 100.00%
```
### Improvement (v3)
```bash
Final Fidelity: 0.9990507302542908
Sequence Length: 81
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.tolerance to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.tolerance` for environment variables or `env.get_wrapper_attr('tolerance')` that will search the reminding wrappers.
  logger.warn(
Successfully approximated the target unitary.
Gate Sequence:
['ryp', 'ryp', 'ryp', 'rzp', 'rzp', 'ryp', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'ryp', 'rxn']
/usr/local/lib/python3.10/dist-packages/gymnasium/core.py:311: UserWarning: WARN: env.gate_set to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.gate_set` for environment variables or `env.get_wrapper_attr('gate_set')` that will search the reminding wrappers.
  logger.warn(
Resultant matrix after applying the gate sequence:
[[ 0.78603125-0.42948813j -0.08707497+0.43601922j]
 [ 0.08707497+0.43601922j  0.78603125+0.42948813j]]
Success rate: 100.00%
```