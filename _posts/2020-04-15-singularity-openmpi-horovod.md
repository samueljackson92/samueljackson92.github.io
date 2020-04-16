---
layout: post
title: OpenMPI, Horovod, and Singularity on HPC
date: 2020-04-16
tags: machine-learning, openmpi, horovod, singularity, tensorflow
published: True
---

Running multi-node, multi-GPU machine learning code on HPC systems is somewhat of a jungle. This post gathers together some lessons learned trying to get Tensorflow 2+ code packaged in a singularity container to work on both multiple nodes & multiple GPUs.

## OpenMPI

OpenMPI is a message passing interface library which can be used to orchestrate code distributed across multiple nodes. It provides an abstraction to facilitate running parallel code without needing to write or understand complexities of networking between different nodes. OpenMPI provides the `mpirun` command to run software written for parallel computing using MPI.

A simple example of running software using `mpirun` is as follows:

```bash
mpirun -np 2 nvidia-smi
```

This runs two processes (using the `-np` flag) and both processes execute the command `nvidia-smi`. This should just print the spec of the GPU twice.

A more complicated example on a cluster of devices might be the following command:

```bash
mpirun -np 2 -H cn2g24:1, cn2g26:1 nvidia-smi
```

This command does exactly the same as the one above except that it runs the same command on two different nodes. The break down of the command is as follows:

- `-np 2` tells OpenMPI to run two processes
- `-H` describes the configuration of how the commands distributed across the cluster. This is a comma separated list of nodes with the number of processes to run on each node
  - In this example `cn2g24` is the node name the `:1` states to run one process on this node.

This should output the results of the `nvidia-smi` from both nodes, but all mixed up.

## Horovod

[Horovod](https://github.com/horovod/horovod) is a framework for distributing deep learning code across multiple nodes. Horovod can integrate into many common deep learning libraries such as Tensorflow and PyTorch. There's a couple of things to note when installing Horovod:

1. You'll need a copy of OpenMPI and `gcc` already installed on your machine
2. You'll need to specify the communication protocol primitives using environment variables that Horovod should use for the broadcasting and all reduce operations during installation. For example  to use the Nvidia's [NCCL communication protocol](https://developer.nvidia.com/nccl) for all reduce and broadcast operations use:

```bash
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install horovod
```

An example of running Horovod is shown below. Some information and a FAQ on debugging Horovod with MPI can be found [here](https://horovod.readthedocs.io/en/latest/mpirun.html).

```bash
mpirun -np 4 												\
	-H cn2g24:2,cn2g26:2 							\
  -bind-to none 										\
  -map-by slot 											\
  -x LD_LIBRARY_PATH 								\
  -x PATH 													\
	-x NCCL_DEBUG=INFO 								\	 
  -x NCCL_SOCKET_IFNAME=^lo,docker0 \
  -mca pml ob1 											\
  -mca btl ^openib 									\
	python train.py
```

Some specific notes on the different parts of this command:

- `-bind-to none` disables bind the processes to a particular core. This is useful is several processes will be running on the same machine.
- `-map-by slot` determines the process mapping strategy to employ to place process. More details on mapping and binding can be found [here](https://www.open-mpi.org/doc/v3.0/man1/mpirun.1.php#sect9).
- `-x` specifies environment variables that should be passed to `mpirun`. In the example this includes the system path and library path. 
  - `NCCL_DEBUG` sets the logging level for NCCL communication protocol.
  - `NCCL_SOCKET_IFNAME` is used to allow or ignore particular network interfaces. The `^` symbol indicates logical NOT and applies tow the whole comma separated list. So in this example we're ignoring both the `docker0`, and `lo` interfaces.
- `-mca`: MCA stands for [Modular Component Architecture](https://www.open-mpi.org/faq/?category=tuning). They are a collection of technologies that are assembled are runtime to produce a concrete MPI implementation. The individual parts are listed below:
  - `pml`: Point-to-point management layer 
  - `ob1`: is one of components in the PML framework which executes communications utilising BTL component
  - `btl`: Byte transfer layer (point-to-point byte movement)
  - `openib`: Open InfiniBand communication layer.

The last two `-mca` flags are a little cryptic, but are basically there to force mpi to run over TCP rather than using [RDMA (Remote Direct Memory Access)](https://en.wikipedia.org/wiki/Remote_direct_memory_access). To quote from the Horovod documentation:

> The `-mca pml ob1` and `-mca btl ^openib` flags force the use of TCP for MPI communication. This avoids many multiprocessing issues that Open MPI has with RDMA which typically results in segmentation faults. Using TCP for MPI does not have noticeable performance impact since most of the heavy communication is done by NCCL, which will use RDMA via RoCE or InfiniBand if they’re available

## Singularity

Singularity is a container technology similar to docker, but more targeted at HPC applications. Applications packaged into singularity containers are distributed as images and provide a self contained set of dependancies for an application. This is perfect for machine learning applications which often have complicated dependancies. 

Executing code packaged as singularity images presents a problem: How do we execute our code *inside* the container on from a different node from *outside* the container. It turns out that Singularity can handle this just fine, but we need to be aware of a couple of things summarised in the [Singularity docs](https://sylabs.io/guides/3.3/user-guide/mpi.html):

> ... when you execute a Singularity container with MPI code, you will call `mpiexec` or a similar launcher on the `singularity` command itself. The MPI process outside of the container will then work in tandem with MPI inside the container and the containerized MPI code to instantiate the job.

So, in other words we need OpenMPI **inside** the container and the version of OpenMPI on the host machine must be compatible with the version **outside** the container. The documentation provides a [starter configuration file](https://sylabs.io/guides/3.3/user-guide/mpi.html) for creating a container with OpenMPI compiled from source inside. Alternatively, you can use the [Nvidia docker images](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) as a base which already have Horovod & OpenMPI installed.

You then need to execute `mpirun` from outside the container:

```bash
mpirun -n <NUMBER_OF_RANKS> singularity exec <PATH/TO/MY/IMAGE> </PATH/TO/BINARY/WITHIN/CONTAINER>
```

## Putting it Together

Putting everything together you should end up with a command similar to the one below. This command runs 4 process in parallel on two different nodes, each with two GPUs. 

1. First we execute `mpirun` with the  various flags we need, which creates `4` processes shared between the two nodes. 
2. Each of the processes starts a new singularity container, using the `horovod.sif` image, which contain an OpenMPI library for communication across nodes and binds the NVIDIA drivers from the host. 
3. Singularity executes the command `python train.py` in the container which runs the machine learning code utilising the Horovod framework.

```bash
mpirun -np 4 -H cn2g24:2,cn2g26:2     
  -bind-to none 										\
  -map-by slot 											\
  -x LD_LIBRARY_PATH 								\
  -x PATH 													\
	-x NCCL_DEBUG=INFO 								\	 
  -x NCCL_SOCKET_IFNAME=^lo,docker0 \
  -mca pml ob1 											\
  -mca btl ^openib 									\
  singularity exec --nv horovod.sif \
  	python train.py
```



