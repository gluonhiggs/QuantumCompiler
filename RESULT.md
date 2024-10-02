# Result

## Fixed target unitary

### Simple fidelity formula (v2)
Original script: [fixed_target_problem_dqn_v2.py](/scripts/fixed_target_problem_dqn_v2.py)
Final Fidelity: `0.991499662833286`

Sequence Length: `70`
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp']
```

Resultant matrix after applying the gate sequence:
```python
[[ 0.8217011-0.37708926j -0.0987385+0.41576641j]
 [ 0.0987385+0.41576641j  0.8217011+0.37708926j]]
```
Success rate: 100.00%

### Simple fidelity formula (v3)
Original script: [fixed_target_problem_dqn_v3.py](/scripts/fixed_target_problem_dqn_v3.py)

Final Fidelity: `0.9990507302542908`
Sequence Length: `81`
Gate Sequence:
```python
['ryp', 'ryp', 'ryp', 'rzp', 'rzp', 'ryp', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'ryp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78603125-0.42948813j -0.08707497+0.43601922j]
 [ 0.08707497+0.43601922j  0.78603125+0.42948813j]]
```
Success rate: 100.00%

### Correct fidelity formula (v1)
Original script: [fixed_target_dqn_v1.py](/scripts/fixed_target_dqn_v1.py)
Resultant matrix after applying the gate sequence:
```python
[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
 ```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the Gate Sequence:
```python
[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Final Fidelity: `0.9900179090775266`
Sequence Length: `67`
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
Final Fidelity: 0.9900179090775266
Sequence Length: 67
 
Gate Sequence:
```python

['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rzp', 'rxn']
```
Resultant matrix after applying the gate sequence:
```python

[[ 0.83613247-0.4020414j  -0.08853904+0.36249972j]
 [ 0.08853904+0.36249972j  0.83613247+0.4020414j ]]
```
Success rate: 100.00%

### Correct fidelity formula (v2)
Original script: [fixed_target_dqn_v2.py](/scripts/fixed_target_dqn_v2.py)
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```

Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Final Fidelity: `0.9993493367877532`
Sequence Length: `77`
 
Gate Sequence:
```python
['rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rxn', 'rzp', 'rxn', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rzp', 'rxn', 'rzp', 'rxn', 'rzp']
```
Resultant matrix after applying the gate sequence:
```python
[[ 0.78619538-0.4207379j  -0.10160515+0.44108143j]
 [ 0.10160515+0.44108143j  0.78619538+0.4207379j ]]
```
Success rate: 100.00%
