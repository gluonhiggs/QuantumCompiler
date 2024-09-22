# Result

## Fixed target unitary

### Improvement (v2)
```bash
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