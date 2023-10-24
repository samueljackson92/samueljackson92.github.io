---
layout: post
title:  How to use SLURM with GPUs
date:   2023-09-28
description: 
tags: SLURM GPU HPC
categories: hpc
---

SLURM is a job scheduler which is designed to fair allocate computational resources amongst many different users. With SLURM you submit a _job_ using a _jobscript_ which specifies which resources you require and how to run your program.

### Basics

In the most basic setup, we want to run our program on a single node for a maximum amount of time. To do this we can write a job script file which looks like the following:

```bash
#SBATCH -n 4
#SBATCH -t 0-00:30
#SBATCH -p centos7
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

python run_my_program.py
```

Parameters for controlling SLURM resources are set using the command `#SBATCH` at the start of the line. These lines should be added at the top of our script before we run our program. Here’s an explanation of what each parameter controls in the script above:

- The `-n` option specifies the number of parallel processes to run. In this case `4`.
- The `-t` option specifies the maximum amount of time that the job will run for. After this time the job will be killed.
- The `-p` option specifies what _partition_ the job should be submitted to. A partition is another name for the _queue_ that this job will enter into. Different partitions can have different resources available, and some partitions may be available only to specific users.
- The `-o` option specifies the name of the file to write any output from our program printed to standard out. The `%j` parameter specifies that the ID of the job should be inserted into the name. For example, if our job was assigned ID `1234` then `output_%j.txt` would translate to `output_1234.txt`.
- The `-e` option specifies the name of the file to write any output from our program printed to standard error.

The final line specifies the program that we wish to run. We can submit our job to the queue (partition) using the `sbatch` command:

```bash
sbatch myjobscript.job
```

### GPUs

If we want to use a GPU, we can request the GPU resources we need using the `gpu` option in our job script. We can write:

 

```bash
...
#SLURM --gpus=2
```

This requests that 2 GPU devices be added to our job. We may also specify the type of GPU we require by adding it before the number of GPUs like so:

 

```bash
#SLURM --gpus=v100:2
```

This requests two GPU devices from the system and specifies that we specifically want each GPU to be a v100 device.

### Multi-Node

For multi-node runs we have to configure:

- The number of nodes
- The number of GPUs per node.

We can add the following lines to our script to specify the resources we need:

```bash
#SLURM -N 4
#SLURM --gpus-per-node=2
...
```

These directives tell SLURM that we want to request four nodes from the system, each with two GPU devices on each node. Therefore we are requesting $4*2 = 8$ GPUs in total. There are other options to control the number of GPUs for your job such as: `--gpus-per-socket`, `--gpus-per-task`, `--mem-per-gpu`, and `--ntasks-per-gpu` . See the [sbatch](https://slurm.schedmd.com/sbatch.html) documentation for more.

### References

- [SLURM sbatch docs](https://slurm.schedmd.com/sbatch.html)
- [Using GPUs with SLURM](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm)
- [Engaging Cluster Docs](https://engaging-web.mit.edu/eofe-wiki/slurm/sbatch/)