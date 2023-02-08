# An End-to-End Reinforcement Learning Approach for Job-Shop Scheduling Problems Based on Constraint Programming

This repository contains the source code for the paper "An End-to-End Reinforcement Learning Approach for Job-Shop Scheduling Problems Based on Constraint Programming".
This works propose an approach to design a Reinforcement Learning (RL) environment using Constraint Programming (CP) 

Check out our HugginFace 🤗 [Space demo](https://huggingface.co/spaces/pierretassel/JobShopCPRL):

<iframe
	src="https://pierretassel-jobshopcprl.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>


## Installation

To use the code, first clone the repository:

```bash
git clone https://github.com/ingambe/End2End-Job-Shop-Scheduling-CP.git
```

It is recommended to create a new virtual environment (optional) and install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Training the Reinforcement Learning Agent

The `main.py` script allows training the agent from scratch:

```bash
python main.py
```

You can train your agent on different instances by replacing the files in the `instances_train/` folder.

The pre-trained checkpoint of the neural network is saved in the `checkpoint.pt` file.

## Solving benchmark instances

The `fast_solve.py` script solves the job-shop scheduling instances stored in the `instances_run/` folder and outputs the results in a `results.csv` file. For better performance, it is recommended to run the script with the `-O` argument:

```bash
python -O fast_solve.py
```

To obtain the solutions using the dispatching heuristics (`FIFO`, `MTWR`, etc.), you can execute the script `static_dispatching/benchmark_static_dispatching.py`

## Looking for the environment only?

The environment only can be installed as a standalone package using

```bash
pip install jss_cp
```

For extra performance, the code is compiled using MyPyC
Checkout the environment repository: [https://github.com/ingambe/JobShopCPEnv](https://github.com/ingambe/JobShopCPEnv)