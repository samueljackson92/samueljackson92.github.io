---
layout: post
title: Parallel Loading with Tensorflow Datasets
---

Tensorflow 2.0 comes with the `tf.data.Dataset` package that provides an easy 
way to ingest and transform data for machine learning pipelines. The easiest way
to create a dataset is directly from existing lists and numpy arrays like so:

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3]) 
for element in dataset: 
  print(element) 
```

Tensorflow `Datasets` can also accept tuples or even dictionaries of arrays as
input as well. For example, imagine you're working on a image segmentation
network which accepts image/mask pairs. The following code creates a dataset
that yields image/mask pairs:

```python
# Generate some random data for testing
# 100 images of size 256x256 with 3 channels
images = np.random.random((100, 256, 256, 3))
# 100 masks of size 256x256 with 1 channel
masks = np.random.randint(0, 1, (100, 256, 256, 1))

dataset = tf.data.Dataset.from_tensor_slices((images, masks))
for image, mask in dataset:
  print(image.shape, mask.shape)
```

This is only useful when the input data will fit fully in memory. If you have
enough data that you cannot possibly fit everything in memory at once then you can load
it directly from storage. For common formats such as [images, CSV data, and text
data](https://www.tensorflow.org/guide/data#reading_input_data) Tensorflow
provides out of the box implementations.

However, this is quite a limited set of supported file types. One way to get
around this limitation is to use the `from_generator` function instead of
`from_tensor_slices`. To use the generator function we must also supply the type
and (optionally) the shape of the returned elements.

```python
%%timeit -n 5 -r 1
def data_generator():
  # Here we're not going to actually read from disk, but we're just going to 
  # return some data with and use a sleep call to simulate file reading
  for _ in range(100):
    # 1 image of size 256x256 with 3 channels
    images = np.random.random((256, 256, 3))
    # 1 mask of size 256x256 with 1 channel
    masks = np.random.randint(0, 1, (256, 256, 1))
    time.sleep(0.1)
    yield images, masks

dataset = tf.data.Dataset.from_generator(data_generator,
                        output_types=(tf.float32, tf.float32))
for image, mask in dataset:
  pass
```

Running this should output something like the following: 

```
>> 5 loops, best of 1: 10.4 s per loop
```

So it takes approximately ~10s to generator 100 image/mask pairs. Intuitively
this makes sense for a single threaded loader. Each image takes approximately
0.1s to load (from the `time.sleep`) and we load 100 of them; so $100 \times 0.1
= 10s$.

However, this solution will not provide optimal performance because it is
inherently **single threaded**. Only one instance of the generator exists and it
will be subject to the python
[GIL](https://wiki.python.org/moin/GlobalInterpreterLock). Tensorflow provides a
nice mechanism to work around this limitation by using the `interleave`
function. This function takes another function, executes that function in
parallel, and "interleaves" the results of each individual call. More info on
`interleave` can be found on
[here](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction).

Below is the same as the previous example, but changed to use `interleave`. The
generator now only yields one image/mask pair, but we now create 100 separate
instances of the generator.

```python
# %%timeit -n 5 -r 1
def data_generator():
  # Here we're not going to actually read from disk, but we're just going to 
  # return some data with and use a sleep call to simulate file reading
  # 1 image of size 256x256 with 3 channels
  images = np.random.random((256, 256, 3))
  # 1 mask of size 256x256 with 1 channel
  masks = np.random.randint(0, 1, (256, 256, 1))
  time.sleep(0.1)
  yield images, masks

def _make_data_generator(x):
  return tf.data.Dataset.from_generator(data_generator,
output_types=(tf.float32, tf.float32))

dataset = (tf.data.Dataset.from_tensor_slices(np.arange(100)) 
            .interleave(_make_data_generator, num_parallel_calls=2))

for image, mask in dataset:
  pass
```

In this code `num_parallel_calls` sets the number of parallel calls to the
generator function to call at the same time. In this example was run on a
machine with two cores, so `num_parallel_calls=2` is a sensible choice. The
output of this code should be something like:

```
5 loops, best of 1: 5.35 s per loop
```

Almost half the time of the first example! It's reasonable to expect a speedup
of 2x as we process two streams of data at the same time.

Additional Reading
---
* [Google Colab Notebook with
  Examples](https://colab.research.google.com/drive/1h9lAtzvHkIRIi0ucNwGbIARiW3b5AVQr)
* [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
* [Building Multi-threaded Custom Data Pipelines for Tensorflow v1.12+](https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5)
* [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance)
