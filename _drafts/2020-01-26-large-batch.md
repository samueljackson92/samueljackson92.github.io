---
layout: post
title: Large Batch Optimizers for Deep Learning
---

* Training CNNs take time
* Brute force solution: add more compute
    * This is relevant for end of Moore's law
    * Train using data parallel SGD
* Scaling no. of workers == scaling batch size.
    * As need more data for each of the workers.
* Increasing batch size means less iterations per epoch.
    * This means you need to take bigger steps in SGD, aka increase the learning
      rate.
* Increasing learning rate makes it more difficult for convergence to happen
    * Networks may diverge, esp. initially.
* Generalisation Gap
    * Large batch converges to sharp minimizers

* One idea: Use warm up with small learning rate, increase after warm up.
* Found that even with this scaling stops at B=2k for AlexNet
* Stability measurements:
    * ratio of normed layer weights to normed gradient magnitude
    * Large == unstable
    * Small == slow training
    * Ratio varies between different layers!
    * Also varies between weights & biases

* Linear LR scaling
    * increase batch size by $k$
    * increase LR by $k$
    * Krizhevsky 2014
    * Hits a limit due to stability issues
    * B=2K for AlexNet on ImageNet.
    * With batch norm we can go higher

* LR Warm-up
    * start with a small LR
    * increase LR after a couple of epochs to usual regime
    * AlexNet B=8K

* Layer-wise Adaptive Rate Scaling (LARS)
    * Differences between ADAM and RMSProp
    * uses different LR for each layer instead of each weight
        * better stability
    * Magnitude of the update is controlled w.r.t the weight norm
    * Implementation:
        * Has global LR
        * Layer wise LR - dependant on trust coefficient: ratio of weights to
          grads.
    * Can train to B=8K with no accuracy loss
    * Can train to B=32K with only 2.6% accuracy loss
    * Training longer with large batch and LARs can result in the same accuracy

* Layer-wise Adaptive Moments Based (LAMB) optimizer
    * Makes some additional changes to LARS
    * Allows for training BERT to more than 32K without performance loss
    * LARS uses SGD as the base algorithm
    * LAMB uses ADAM as the base algorithm
        * Per dimension normalisation of w.r.t the second moment as in ADAM
        * Layer-wise normalisation

* Layer-wise Adaptive Rate Clipping (LARC)
    * Builds on LARS to include an option to clip or scale
    * Superset of LARS to clip or scale
    * Implementation in APEX for PyTorch

Additional Reading
---

* Paper introducing LARS - [Large Batch Training of Convolutional
  Networks](https://arxiv.org/abs/1708.03888)
* Paper introducing LAMB - [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/pdf/1904.00962.pdf)
* NVIDIA's description of LARC - [Pretraining BERT with Layer-wise Adaptive Learning Rates](https://devblogs.nvidia.com/pretraining-bert-with-layer-wise-adaptive-learning-rates/)
* [TF 2.0 Implementation of LAMB](https://github.com/tensorflow/addons)
