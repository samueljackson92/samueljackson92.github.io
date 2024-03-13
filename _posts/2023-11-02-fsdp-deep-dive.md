---
layout: post
title: A Deep Dive into Fully Sharded Data Parallelism
date: 2023-11-02
description: 
tags: Scaling ML GPU HPC
categories: hpc
---

In the world of deep learning, the size of neural network models has been growing exponentially. With models like GPT-3 boasting a staggering 175 billion parameters, it has become essential to find efficient ways to train these large models while managing the significant memory requirements.

[Fully Shared Data Parallelism](http://arxiv.org/abs/2304.11277) (FSDP) is a novel approach to address the challenges of training large neural network models. Traditionally, data parallelism has been a popular method for training neural network models. However, it falls short when it comes to handling large models effectively. Other techniques like pipeline parallelism and tensor parallelism have been explored, but they often lack a generic solution that can be applied across different models.

## Background

In this section, we briefly review the concepts of model replication, partitioning, and sharding which FSDP builds upon. 

### Model replication

Model replication is a technique that is specifically designed to handle datasets with high volumes of data. It achieves this by scaling out computations across different devices, ensuring efficient processing.

One method of model replication is data parallelism. This involves maintaining a replica of the model on each device and synchronizing them through 'allreduce' operations during the backward pass. Additionally, 'DDP' (Distributed Data Parallel) allows for overlapping communications with backward computation. However, it is important to note that all replicas must fit within a single GPU, which is a limitation.

### Model partitioning

When models are too large to fit within a single GPU, model partitioning becomes necessary. This approach involves dividing the model into multiple parts and distributing them across different GPUs. Two techniques that fall under this category are pipeline parallelism and tensor parallelism.

Pipeline parallelism breaks the model into a sequence of layers. However, it is limited to a sequence of layers and can become a communication bottleneck.

On the other hand, Tensor parallelism offers a lower-level approach by breaking layers into smaller parts and distributing them. Implementing tensor parallelism typically requires modification of the source code to annotate how to break apart and reconstruct tensors.

### Model Sharding

Sharding parameters can be an effective way to reduce the memory footprint of a model. Sharding a model breaks layer into several parts, with each rank only holding a portion of the model parameters. During forward and backward passes, the parameters are materialised in sequence on each device.

There two approaches distinct approaches to creating sharded models. 

1. The first approach involves performing computations locally with just the parameter shards and communicating resulting activations. However, this requires communication in the "critical path" of the model which limits the ability to interlace communication.
2. The second approach involves communicating the parameters, so that every rank has a full copy of the whole layer. The advantage of this approach is that parameter communications do not a communication dependency in the critical path and can be overlapped.

FSDP is an example of the second category, where the parameter shards are communicated in order to replicate the whole layer on each device. 

## Algorithm Overview

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19e30c16-0be2-42b8-b809-bd54af0c7474/89340dd2-2ef3-48b6-8e0a-9e761e509f15/Untitled.png)

FSDP decomposes the model into smaller *units*, known as shards, by splitting the parameters for each layer. Each unit may consist of more than one layer. For example in figure 1 from the paper, layer 1 and layer 2 are grouped into the same unit. This flexible design allows a model to be split to different levels of granularity. 

Each shard is then distributed to a different device, such that only part of the fully model is held by each GPU. During a forward pass, the units are materialized one at a time, meaning that they are created and processed sequentially. An overview of this process is given in the diagram below.

![fsdp (2).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/19e30c16-0be2-42b8-b809-bd54af0c7474/18d6359e-4d66-41e2-996b-0736c744d390/fsdp_(2).png)

In the example above, our model consists of a simple MLP of three layers (pink, blue, and orange). Each layer is split into two shards (coloured white and black).  Each shard is then distributed to one of two GPUs represented by the black boxes. Notice that in this toy example, each GPU does not have enough space to hold a full replica of the model.

When we perform a forward pass, FSDP first materialises the whole of the first layer on each GPU by communicating is layer #1 shard (pink) with the other device. Once the forward pass has been performed, the materialised shards are discarded. Then the algorithm proceeds to materialize the next layer (blue), and repeats the process.

Through this mechanism, it is possible for a model to be larger than the memory of a single device while still having each device operate on a full replica of each model by materializing each layer one by one during forward and backwards passes.

**Note:** importantly the sharding performed by FSDP does not need to be at the level of a single layer, but at the *unit* level. What constitutes a unit within the model is left to the user to decide. In the pytorch implementation this is controlled using [auto-sharding policies](https://pytorch.org/docs/stable/fsdp.html).

### Model Initialisation

Sharded model creation can be challenging due to two key reasons:

1. The difficulty lies in creating a model instance without allocating any tensor storage. Typically, tensor storage is allocated upon creation. To overcome this, a solution is to implement deferred initialization. This involves creating a fake device, known as **`device="meta"`**, which does not consume any memory but appears as if it does in PyTorch.
2. Another challenge is ensuring the proper initialization of model parameters in line with the user's implementation. This can be addressed by sequentially initializing the parameters, one unit at a time. This process involves materializing the unit, initializing it, and then unmaterializing it.

## Sharding Strategies

FSDP provides two main different types of sharding strategy: Full Sharding and Hybrid Sharding. Sharding strategies play a significant role in determining the memory footprint and communication overhead of a system. The difference between Full and Hybrid sharding can be described by considering the sharding factor $$F$$.

Formally, If we have $$N$$ number of devices, the sharding factor, $$F$$, indicates the number of ranks over which parameters are divided. It determines the granularity of sharding and affects the distribution of parameters across devices.

Let's consider different sharding strategies:

- When $$F = 1$$, it means full replication, equivalent to DDP (Distributed Data Parallelism). In this strategy, each device holds a full replica of the model. While it guarantees no communication overhead, it comes with a high memory overhead due to the replication of parameters on each device.
- On the other hand, when $$F = N$$, the model will be fully sharded. In this strategy, each device holds $$\frac{1}{W}$$ of the model parameters, where $$W$$ is the total number of shards. This approach minimizes the memory overhead as each device only needs to store a portion of the parameters. However, it introduces a higher communication cost since the devices need to communicate and synchronize their parameter updates during training.
- Hybrid sharding occurs when $$1 < F < N$$. In this strategy, the parameters are divided into groups, and each shard is replicated across multiple devices. This approach allows for a balance between memory overhead and communication cost. It can also be used to mimic network topology, minimizing cross-host traffic in distributed training settings.

Now, let's delve into the details of fully sharding and hybrid sharding strategies:

### Full Sharding

This strategy offers the lowest memory overhead but comes with the highest communication cost. Compared to DDP, the FSDP paper estimated that fully sharded models have a 1.5x higher communication overhead and volume.

This is caused by inefficiencies in training may arise due to AllReduce operations with uneven tensor sizes. NCCL, a popular communication library, requires even tensor input sizes for efficient communication.

- To address this issue, the use of a FlatParameter is recommended. A FlatParameter is a 1D tensor constructed by concatenating the flattened parameters and adding right padding to achieve an even size. This allows for efficient communication and avoids the need for unnecessary copies.
- The FlatParameter is then divided into $F$ chunks, where each chunk is assigned to a specific rank or device. This approach enables arbitrary tensor shapes while minimizing padding.
- The exact layout required by AllGather and ReduceScatter operations is achieved, ensuring efficient communication and avoiding unnecessary data copies.

### Hybrid Sharding

- In this strategy, the parameters are divided into groups or shards, and each shard is replicated across devices.
- Hybrid sharding can be used to mimic network topology, where each shard represents a portion of the model that corresponds to a specific network component or layer.
- By replicating the shards across devices, the communication cost is reduced compared to fully sharded models, as the communication is limited within each group of devices.
- This strategy can be particularly useful in distributed training settings where minimizing cross-host traffic is essential for efficient training.

Overall, sharding strategies play a crucial role in managing the memory requirements and communication overhead when training large neural network models. The choice of the sharding factor $F$ depends on the specific model, available resources, and desired trade-offs between memory usage and communication cost. Understanding these strategies can help optimize the training process and improve the efficiency of training large-scale models.
