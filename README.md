# QuantumCompiler

This is a repository for the Quantum Compiler project. The goal of this project is to **reproduce the results** of the paper [Quantum compiling by deep reinforcement learning](https://www.nature.com/articles/s42005-021-00684-3#ref-CR7). The paper presents a quantum compiler that uses deep reinforcement learning to optimize the compilation of quantum circuits.

## Problems

The paper presents the following problems:

[x] Training neural networks for approximate a single-qubit gate

[] Quantum compiling by rotation operators

[] Quantum compiling by the HRC efficiently universal base of gates

Note: [x] means the problem has been solved.
## Installation

Clone the repository and open it with VSCode. Press `Ctrl+Shift+P` on Windows or `Cmd+Shift+P` on MacOS to open the command palette. Then type `Open the Folder in Container` and select the repository folder. This will open the repository in a Docker container with all the necessary dependencies installed.

## Usage

Run the following command to run a script in the container:

```bash
python3 <script_name>.py
```

For problem 1, run the following command:

```bash
python3 fixed_target_problem_dqn_v3.py
```
Similarly for other problems. You can try to run different scripts to see how things work.

## Authors

Duong H. D. Tran