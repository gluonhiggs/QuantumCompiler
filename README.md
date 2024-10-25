# QuantumCompiler

This is a repository for the Quantum Compiler project. 

The initial goal of this project is to **reproduce the results** of the paper [Quantum compiling by deep reinforcement learning](https://www.nature.com/articles/s42005-021-00684-3#ref-CR7). The paper presents a quantum compiler that uses deep reinforcement learning to optimize the compilation of quantum circuits.

With the success of reproducing the results, we (which is me ) will extend the project to solve more problems. The next target is to explore what we can do with this paper [Quantum Compiling with Reinforcement Learning on a Superconducting Processor](https://arxiv.org/pdf/2406.12195)

## Problems

I'm going to solve following problems:

[x] Training neural networks for approximate a single-qubit gate

[x] Quantum compiling by rotation operators

[] Quantum compiling by rotation operators using singular value constraints

[] Quantum compiling by the HRC efficiently universal base of gates

[] Generalization to multi-qubit gates

Note: [x] means the problem has been solved.
## Setup and Usage

### Pre-requisites

- [Docker](https://www.docker.com/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed in VSCode
- [Git](https://git-scm.com/)
- You don't need to install Python, CUDA Toolkit or any other dependencies. Everything is provided in the Docker container. 

### Setup

Clone the repository and open it with VSCode. Press `Ctrl+Shift+P` on Windows or `Cmd+Shift+P` on MacOS to open the command palette. Then type `Open the Folder in Container` and select the repository folder. This will open the repository in a Docker container with all the necessary dependencies installed.

### Usage

Run the following command to run a script in the container:

```bash
python3 <script_name>.py
```

For problem 1, run the following command:

```bash
python3 scripts/fixed_target_problem_dqn_v3.py
```
Similarly for other problems. You can try to run different scripts to see how things work.

## Results

Training progress plots for each script are stored in the `data/` directory. Logs and results are captured and summarized to [RESULTS.md](RESULT.md) file.

**If you find this project helpful, please consider giving it a ⭐. I appreciate that!**

## Authors

Duong H. D. Tran
