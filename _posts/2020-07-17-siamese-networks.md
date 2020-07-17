---
layout: post
title: Siamese Networks in PyTorch
---

Siamese networks are a classic method to perform one-shot image classification. They were originally proposed in [this paper by Koch, Zemel & Salakhutdinov (2015)](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf). Siamese networks are a form of metric learning, which is a subset of meta-learning. See [this blog post](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#define-the-meta-learning-problem) for a nice overview of networks in these areas.

In siamese networks, instead of learning how to classify inputs according to a particular label, *how to differentiate between examples from different classes*. As a concrete example, in the MNIST digit classification problem, we normally learn to classify the digits into ten categories using a CNN with softmax on the final layer. Our output is then vector of size 10 which indicates which class the digit belongs to.

On the other hand, in metric learning we instead learn to distinguishes the different classes by showing the network different sets of positive and negative pairs of examples. In the example below we feed pairs of images through an identical network. The pairs of examples can either be positive (i.e. the examples belong to the same class) or negitive (i.e. the examples belong to different classes). By using a metric (such as absolute difference) between the two learned representations, we can train the network to distinguish between classes.

![](/assets/images/2020-07-17/siam.png)

The rest of this post demonstrates how to implement a Siamese network to classify images from the MNIST dataset using PyTorch.

## PyTorch Implementation

### Network Implementations

We first need to define a "backbone" network. Here we just going to use a very simple CNN. The siamese network contains the CNN and takes as input both "left" batch and a "right" batch. The left and right batches should contain pairs of positive or negative pairs of images. To score the similarity between the two images we take the absolute difference between their representations and then pass this though a sigmoid layer, giving us a score between 0-1.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    """ A simple CNN"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SiameseNet(nn.Module):
    """A siamese network"""
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.net = CNN()
        
        self.fc1 = nn.Linear(84, 1)
        
    def forward(self, left_batch, right_batch):
        """Perform a forward pass of the siamese network
        
        Args:
            left_batch: images to pass along the "left" network path
            right_batch: images to pass along the "right" network path
        """
        left_output = self.net.forward(left_batch)
        right_output = self.net.forward(right_batch)
        
        diff = torch.abs(left_output - right_output)
        probs = F.sigmoid(self.fc1(diff))
        return probs
```

### Data Loader

The most complicated part of a siamese network is actually not the network itself, but setting up the positive and negative pairs for training. This code is based on the code in [this blog post](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d). In the code below:

 - Load the MNIST dataset
 - For each batch:
  - Randomly choose a class for each index in the batch
  - For the first half of the batch
      - Randomly select negative images from *different* classes
  - For the second half of the batch
      - Randomly select positive images from the *same* class
      
It is important to make sure that the batch contains an equal number of positive and negitive examples so that we don't run into problems with class imbalance.


```python
class MNISTOneShotDataLoader(Dataset):
    """Data loader for generating batches of image pairs for one-shot learning."""

    def __init__(self, file_path, batch_size, n_batches=10, transform=None):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            batch_size (int): size of the batches to generate
            n_batches (int): number of batches to generate at each epoch
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(file_path, header=None)
        self.labels = df[0]
        mnist_df = df.drop(0, axis=1)
        self.n_samples, _ = mnist_df.shape 

        mnist = mnist_df.values.reshape((self.n_samples, 28, 28))
        classes = np.unique(self.labels)

        self.mnist = [mnist[self.labels == c] for c in classes]
        self.num_classes = len(classes)

        self.batch_size = batch_size
        self.transform = transform
        self.n_batches = n_batches
        
    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if idx > self.n_batches:
            raise StopIteration()

        batch, targets = self.get_batch(self.labels, self.batch_size)
        sample = {'images': batch, 'targets': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_batch(self, labels, batch_size):
        """
        Create batch of n pairs, half same class, half different class
        """
        n_classes = len(np.unique(labels))
        w, h = 28, 28

        # randomly sample several classes to use in the batch
        categories = np.random.choice(n_classes, size=(batch_size,),replace=True)

        # initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]

        # initialize vector for the targets
        targets=np.zeros((batch_size,))
        # make one half of it '1's, so 2nd half of batch has same class
        targets[batch_size//2:] = 1

        for i in range(batch_size):
            category = categories[i]

            idx_1 = rng.randint(0, len(self.mnist[category]))
            pairs[0][i,:,:,:] = self.mnist[category][idx_1].reshape(w, h, 1)

            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category  
            else: 
                # add a random number to the category modulo n classes to ensure 2nd image has a different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes

            idx_2 = rng.randint(0, len(self.mnist[category_2]))
            pairs[1][i,:,:,:] = self.mnist[category_2][idx_2].reshape(w, h, 1)

        return pairs, targets

class ToTensorTransform(object):
    """Helper transform to convert numpy images to pytorch tensors """
    def __call__(self, batch):
        inputs, targets = batch['images'], batch['targets']
        left, right = inputs
        
        left = torch.from_numpy(left).float()
        left = left.permute(0, 3, 1, 2)
        
        right = torch.from_numpy(right).float()
        right = right.permute(0, 3, 1, 2)

        targets = torch.from_numpy(targets).float()
        
        return {'images': (left, right), 'targets': targets}
```

We can plot a batch of images and check that the positive and negative pairs are sensible


```python
data_loader = MNISTOneShotDataLoader('data/mnist_train.csv', 10)

data = data_loader[0]
batch = data['images']

fig, ax = plt.subplots(2, 10, figsize=(20, 5))
for i in range(batch_size):
    label = 'same' if data['targets'][i] == 1 else 'different'
    ax[0][i].set_title(label)
    ax[0][i].matshow(batch[0][i, ..., 0])
    ax[1][i].matshow(batch[1][i, ..., 0])
    ax[0][i].axis('off')
    ax[1][i].axis('off')
```


![png](/assets/images/2020-07-17/output_6_0.png)


### Training

Now we're ready to train the network. This is relatively simple to setup. We just need to create an instance of the network, train and test data loaders and setup the optimizer. As the output of our network is a similarity score between 0 and 1 we can train using standard binary cross-entropy.


```python
from sklearn.metrics import accuracy_score

# Setup data loaders
data_loader = MNISTOneShotDataLoader('data/mnist_train.csv', 100, n_batches=1000, transform=ToTensorTransform())
test_data_loader = MNISTOneShotDataLoader('data/mnist_test.csv', 100, n_batches=1000, transform=ToTensorTransform())

# Setup network
net = SiameseNet()
net.to(device)
net.train()


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
criterion = nn.BCELoss()

for epoch in range(5): 
    acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['images'], data['targets']
        left, right = inputs

        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        probs = net.forward(left, right)
        loss = criterion(probs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        acc += accuracy_score(probs.data.cpu() > .5, labels.data.cpu())
        
        if i % 100 == 0:
            print('[%d, %5d] loss: %.6f accuracy: %.2f' %
                  (epoch + 1, i + 1, running_loss / 100, acc / 100))
            running_loss = 0.0
            acc = 0.0

print('Finished Training')
```

    [1,     1] loss: 0.008948 accuracy: 0.01
    [1,   101] loss: 0.614116 accuracy: 0.67
    [1,   201] loss: 0.467475 accuracy: 0.78
    [1,   301] loss: 0.412342 accuracy: 0.82
    [1,   401] loss: 0.373413 accuracy: 0.84
    [1,   501] loss: 0.342862 accuracy: 0.85
    [1,   601] loss: 0.323661 accuracy: 0.87
    [1,   701] loss: 0.299436 accuracy: 0.88
    [1,   801] loss: 0.288102 accuracy: 0.89
    [1,   901] loss: 0.267748 accuracy: 0.89
    [1,  1001] loss: 0.247009 accuracy: 0.90
    [2,     1] loss: 0.002207 accuracy: 0.01
    [2,   101] loss: 0.240778 accuracy: 0.91
    [2,   201] loss: 0.227923 accuracy: 0.91
    [2,   301] loss: 0.221650 accuracy: 0.92
    [2,   401] loss: 0.214739 accuracy: 0.92
    [2,   501] loss: 0.206456 accuracy: 0.92
    [2,   601] loss: 0.193130 accuracy: 0.93
    [2,   701] loss: 0.195499 accuracy: 0.93
    [2,   801] loss: 0.192836 accuracy: 0.93
    [2,   901] loss: 0.180753 accuracy: 0.93
    [2,  1001] loss: 0.183278 accuracy: 0.93
    [3,     1] loss: 0.001035 accuracy: 0.01
    [3,   101] loss: 0.166679 accuracy: 0.94
    [3,   201] loss: 0.169835 accuracy: 0.94
    [3,   301] loss: 0.176420 accuracy: 0.93
    [3,   401] loss: 0.158579 accuracy: 0.94
    [3,   501] loss: 0.160942 accuracy: 0.94
    [3,   601] loss: 0.159728 accuracy: 0.94
    [3,   701] loss: 0.158514 accuracy: 0.94
    [3,   801] loss: 0.156543 accuracy: 0.94
    [3,   901] loss: 0.154978 accuracy: 0.94
    [3,  1001] loss: 0.148229 accuracy: 0.94
    [4,     1] loss: 0.001153 accuracy: 0.01
    [4,   101] loss: 0.151807 accuracy: 0.94
    [4,   201] loss: 0.143181 accuracy: 0.95
    [4,   301] loss: 0.143988 accuracy: 0.95
    [4,   401] loss: 0.137953 accuracy: 0.95
    [4,   501] loss: 0.135806 accuracy: 0.95
    [4,   601] loss: 0.136655 accuracy: 0.95
    [4,   701] loss: 0.131614 accuracy: 0.95
    [4,   801] loss: 0.132453 accuracy: 0.95
    [4,   901] loss: 0.130402 accuracy: 0.95
    [4,  1001] loss: 0.128495 accuracy: 0.95
    [5,     1] loss: 0.001804 accuracy: 0.01
    [5,   101] loss: 0.123973 accuracy: 0.96
    [5,   201] loss: 0.124438 accuracy: 0.96
    [5,   301] loss: 0.126210 accuracy: 0.95
    [5,   401] loss: 0.124390 accuracy: 0.96
    [5,   501] loss: 0.128193 accuracy: 0.95
    [5,   601] loss: 0.114783 accuracy: 0.96
    [5,   701] loss: 0.117170 accuracy: 0.96
    [5,   801] loss: 0.117277 accuracy: 0.96
    [5,   901] loss: 0.117696 accuracy: 0.95
    [5,  1001] loss: 0.115676 accuracy: 0.96
    Finished Training


### Testing

Finally we can look at the predicted classification of some samples from the test set. We can see that we can accurately differentiate between classes.


```python
data = test_data_loader[0]
batch = data['images']

left, right = batch
left = left.to(device)
right = right.to(device)
        
probs = net.forward(left, right)

fig, ax = plt.subplots(2, 10, figsize=(20, 5))
for i in range(batch_size):
    label = 'same' if probs[i*10] > .5 else 'different'
    ax[0][i].set_title(label)
    ax[0][i].matshow(batch[0][i*10, 0].data.cpu())
    ax[1][i].matshow(batch[1][i*10, 0].data.cpu())
    ax[0][i].axis('off')
    ax[1][i].axis('off')
```


![png](/assets/images/2020-07-17/output_10_0.png)

