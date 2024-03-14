# Benchmark on saved models
There is a high likelihood that you, at some point, find yourself wanting to benchmark models that were trained previously.
In this guide we will walk through how we can accomplish this with nnbench.

## Example: Named Entity Recognition
We will start with an aside that talks through the setup of the example we will use in this guide.
If you are only interested in the application of nnbench, you can skip this section.

There are lots of reasons why you could want to retrieve saved models for benchmarking. 
Among them these are reviewing the work of colleagues, comparing experimental performances to an existing benchmark, or dealing with models that require significant compute such that in-place retraining is impractical.
For this example, we deal with a named entity recognition (NER) model that is based on the pre-trained encoder-decoder transformer [BERT](https://arxiv.org/abs/1810.04805).
The model is trained on the [CoNLLpp dataset](https://huggingface.co/datasets/conllpp) which consists of sentences from news stories where words were tagged with Person, Organization, Location, or Miscellaneous if they referred to entities. 
Words are assigned an out-of-entity label if they do not represent an entity.

### Model Training
You find the code to train the model in the nnbench [repository](https://github.com/aai-institute/nnbench) in the directory `examples/artifact_benchmarking/src/training/training.py`.
If you want to skip running the training script but still want to reproduce this example, you can take any BERT model fine tuned for NER with the CoNLL dataset family.
You find many on the Huggingface model hub, for example [this one](https://huggingface.co/dslim/bert-base-NER). You need to download the `model.safetensors`, `config.json`, `tokenizer_config.json`, and `tokenizer.json` files.
If you want to train your own model, continue below. 

There is some necessary preprocessing and data wrangling to train the model. 
We will not go into the details here. But if you are interested in a more thorough walkthrough, look into this [resource](https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt) by Huggingface which served as the basis for this example. 
It is not feasible to train the model on a CPU. If you do not have access to a GPU you can use free GPU instances on [Google Colab](https://colab.research.google.com/).
When you open a new notebook there make sure to select a GPU instance in the upper right corner.
The you can upload the `training.py`.
You can ignore An eventual warning that the data is not persisted.
Next, install the necessary dependencies: `!pip install datasets transformers[torch]`.
Google Colab comes with some dependencies already installed in the environment.
Hence, if you are working with a different GPU instance, make sure to install everything from the `pyproject.toml` in the `examples/artifact_benchmarking` folder. 
Next you can execute the `training.py` with `!python training.py`.
This will train two BERT models ("bert-base-uncased" and "distilbert-base-uncased") which we can compare using nnbench. 
If you want, you can adapt the training script to train other models by editing the tuples in the `tokenizers_and_models` list at the bottom of the training script. 
The training of the models takes around 10 minutes.
Once it is done, download the respective files and save them to your disk.
They should be the same mentioned above. 
We will need the path to the files for benchmarking later.

### The Benchmarks
The benchmarking code is found in the `examples/artifact_benchmarking/benchmark.py`.
We calculate precision, recall, accuracy, and f1 scores for the whole test set and specific labels.
Additionally, we obtain information about the model such as its memory footprint and inference time.
We are not walking through the whole file but instead point out certain design choices as an inspiration to you. 
If you are interested in a more detailed walkthrough on how to set up benchmarks, you can find it [here](../guides/benchmarks.md).
Notable design choices in this benchmark are that we factored out the evaluation loop as it is necessary for all evaluation metrics. We cache it using the `functools.lru_cache` decorator so the evaluation loop runs only once per benchmark run instead of once per metric which greatly reduces runtime.
We use `nnbench.parametrize` to get the per-class metrics. 
As the parametrization method needs the same arguments for each benchmark, we use Python's builtin `functools.partial` to fill the arguments.
One noteworthy subtlety here is that we need to call the partial immediately so it returns the pre-filled `nnbench.parametrize` decorators.
If we don't do that, the `runner.collect` does not find the respective benchmarks. 

## Running Benchmarks with saved Artifacts
Now that we have explained the example, let's jump into the benchmarking code.
You find it in the nnbench repository in `examples/artifact_benchmarking/src/runner.py`.

The benchmarking code is written to be executed as a script that consumes any number of file paths to models to benchmark as arguments, `python runner.py /path/to/model1 /path/to/model2`.
The parsing of the arguments is handled by the last lines in the script which then calls the `main()` function:

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:91:95"
```

### Artifact Classes
The `main()` function first sets up the benchmark reporter and the runner. 
Then we create a list of models. These models are instances of a `TokenClassificationModel`, a custom class we implemented which inherits from the `Artifact` base class.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:30:37"
```

The `Artifact` class is a typesafe wrapper around serialized data of any kind.
It allows for lazy deserialization of artifacts from a  path attribute.
This attribute is set by the `ArtifactLoader` (which we will cover in a moment) that supplied a path to the local artifact disk storage. 
In our derived class, we have to override the `deserialize()` method to properly load the artifact value into memory.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:56:57"
```

The `deserialize()` method has to set the `self._value` attribute to the value we want to access later.
In this case, we assign it a tuple containing the Huggingface Transformers model and tokenizer.

We do similar with the CoNLLpp dataset.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:30:37"
```

In this case, we store the `datasets.Dataset` object as well ass a dictionary which maps the label id to a semantic string in the `_value` attribute. 
The value that we store in the `_value` of an artifact can be of any kind and that we use tuples in both instances here is a circumstance.

### Artifact Loaders
Upon instantiation of an `Artifact` or derived classes we need to supply an `ArtifactLoader` (or a class derived from it). `ArtifactLoader`s are classes that implement a `load()` method that resolves to a path-like object or string which points to the local storage location of the artifact. This method is used by the Artifact class. 

For our models, we use the provided `LocalArtifactLoader` which consumes a path and passes it on later.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:52:54"
```

We have a little more logic with respect to the dataset as we handle the train test split as well.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:19:27"
```

Now we execute the benchmark in the loop over the different models.

```python
--8<-- "examples/artifact_benchmarking/src/runner.py:59:88"
```
