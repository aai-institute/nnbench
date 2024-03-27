# Benchmarking HuggingFace models on a dataset
There is a high likelihood that you, at some point, find yourself wanting to benchmark previously trained models.
This guide shows you how to do it for a HuggingFace model with nnbench.

## Example: Named Entity Recognition
We start with a small tangent about the example setup that we will use in this guide.
If you are only interested in the application of nnbench, you can skip this section.

There are lots of reasons why you could want to retrieve saved models for benchmarking. 
Among them these are reviewing the work of colleagues, comparing model performance to an existing benchmark, or dealing with models that require significant compute such that in-place retraining is impractical.

For this example, we look at a named entity recognition (NER) model that is based on the pre-trained encoder-decoder transformer [BERT](https://arxiv.org/abs/1810.04805) from HuggingFace.
The model is trained on the [CoNLLpp dataset](https://huggingface.co/datasets/conllpp) which consists of sentences from news stories where words were tagged with Person, Organization, Location, or Miscellaneous if they referred to entities. 
Words are assigned an out-of-entity label if they do not represent an entity.

## Model Training
You find the code to train the model in the nnbench [repository](https://github.com/aai-institute/nnbench/tree/main/examples/huggingface).
If you want to skip running the training script but still want to reproduce this example, you can take any BERT model fine tuned for NER with the CoNLL dataset family.
You find many on the Huggingface model hub, for example [this one](https://huggingface.co/dslim/bert-base-NER). You need to download the `model.safetensors`, `config.json`, `tokenizer_config.json`, and `tokenizer.json` files.
If you want to train your own model, continue below.

There is some necessary preprocessing and data wrangling to train the model. 
We will not go into the details here, but if you are interested in a more thorough walkthrough, look into this [resource](https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt) by Huggingface which served as the basis for this example.

It is not feasible to train the model on a CPU. If you do not have access to a GPU, you can use free GPU instances on [Google Colab](https://colab.research.google.com/).
When opening a new Colab notebook, make sure to select a GPU instance in the upper right corner.
Then, you can upload the `training.py`. You can ignore any warnings about the data not being persisted.

Next, install the necessary dependencies: `!pip install datasets transformers[torch]`.  Google Colab comes with some dependencies already installed in the environment.
Hence, if you are working with a different GPU instance, make sure to install everything from the `pyproject.toml` in the `examples/artifact_benchmarking` folder.

Finally, you can execute the `training.py` with `!python training.py`.
This will train two BERT models ("bert-base-uncased" and "distilbert-base-uncased") which we can compare using nnbench. 
If you want, you can adapt the training script to train other models by editing the tuples in the `tokenizers_and_models` list at the bottom of the training script. 
The training of the models takes around 10 minutes.

Once it is done, download the respective files and save them to your disk.
They should be the same mentioned above. We will need the paths to the files for benchmarking later.

## The benchmarks

The benchmarking code is found in the `examples/huggingface/benchmark.py`.
We calculate precision, recall, accuracy, and f1 scores for the whole test set and specific labels.
Additionally, we obtain information about the model such as its memory footprint and inference time.

We are not walking through the whole file but instead point out certain design choices as an inspiration to you. 
If you are interested in a more detailed walkthrough on how to set up benchmarks, you can find it [here](../guides/benchmarks.md).

Notable design choices in this benchmark are that we factored out the evaluation loop as it is necessary for all evaluation metrics.
We cache it using the `functools.cache` decorator so the evaluation loop runs only once per benchmark run instead of once per metric which greatly reduces runtime.

We also use `nnbench.parametrize` to get the per-class metrics.
As the parametrization method needs the same arguments for each benchmark, we use Python's builtin `functools.partial` to fill the arguments.

```python
--8<-- "examples/huggingface/benchmark.py:131:139"
```

!!! Tip
    In this parametrization, the model path is hardcoded to "dslim/distilbert-NER" on the HuggingFace hub.
    When benchmarking other models, be sure to change this path to the actual model you want to benchmark.

After this, the benchmarking code is actually very simple, as in most of the other examples.
You find it in the nnbench repository in `examples/huggingface/runner.py`.

## Custom memo classes

The parametrization contains a list of models, which are each instances of a `TokenClassificationModelMemo` a custom class we implemented which inherits from the `nnbench.Memo` class.
A big advantage of a memo in this case is its ability to lazy-load models and later evict the loaded models again from a cache.

```python
--8<-- "examples/huggingface/benchmark.py:23:35"
```

The `Memo` class is a generic wrapper around serialized data of any kind.
It allows for lazy deserialization of artifacts from uniquely identifying metadata like storage paths, checksums, or model names on HuggingFace Hub in our case.
In our derived class, we have to override the `Memo.__call__()` method to properly load the memoized value into memory.

We do similar with the CoNLLpp dataset.

```python
--8<-- "examples/huggingface/benchmark.py:51:64"
```

In this case, we lazy-load the `datasets.Dataset` object.
In the following `IndexLabelMapMemo` class, we store a dictionary mapping the label ID to a semantic string.

```python
--8<-- "examples/huggingface/benchmark.py:67:82"
```

!!! Info
    There is no need to type-hint `TokenClassificationModelMemo`s in the corresponding benchmarks -
    the benchmark runner takes care of filling in the memoized values for the memos themselves.

Because we implemented our memoized values as four different memo class types, this modularizes the benchmark input parameters -
we only need to reference memos when they are actually used. Considering the recall benchmarks:

```python
--8<-- "examples/huggingface/benchmark.py:174:204"
```

we see that the memoized `index_label_mapping` argument is only necessary in the per-class benchmark, so it is never passed to the main computation.

!!! Tip
    When implementing memos for a benchmark workload, using only one value per memo at the cost of another class definition is often worth it,
    since you have more direct control over what goes into your benchmarks, and you can avoid having unused parameters altogether with this approach.
