# Using artifacts in nnbench
With more complex benchmarking set-ups you will find yourself wanting to use static artifacts.
These can be, for example, test and validation data, or serialized model files from a model registry.
nnbench provides an artifact framework to handle these assets.
This framework consists of the `ArtifactLoader`, the `Artifact`, and the `ArtifactCollection` base classes.

Conceptually, they are intended to be used as follows:
- `ArtifactLoader` to load the artifact onto the local filesystem,
- `Artifact` to handle the artifact within nnbench and enable lazy loading into memory,
- `ArtifactCollection` as a list wrapper around artifacts to enable iterations over artifacts.

You can implement your own derivative classes to handle custom logic for artifact deserialization and loading.
Additionally, we provide some derived classes out of the box to handle e.g. local filepaths or filesystems which are covered by pythons fsspec package.
Let us now discuss each class of the framework in detail.

## Using the `ArtifactLoader`
The `ArtifactLoader` is an abstract base class for which you can implement your custom instance by overriding the `load()` method which needs to return a filepath either as a `PathLike` object or a string.
You can see an example on how that is implemented in the implementation of the `LocalArtifactLoader` (that is also provided out of the box by nnbench).
```python
import os
from pathlib import Path
from nnbench.types import ArtifactLoader

class LocalArtifactLoader(ArtifactLoader):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = path

    def load(self) -> Path:
        return Path(self._path).resolve()
```

The use of these `ArtifactLoader` becomes apparent when you think about using it for remote artifact storage locations such as an S3 bucket. 
Besides the barebones `LocalArtifactLoader`, nnbench also provides the `nnbench.types.FilePathArtifactLoader`.
To use it you have to `pip install fsspec` as an additional dependency.
The class is then able to handle different filepath formats such as S3 links (or links to lakeFS, should you have [lakeFS-spec](https://lakefs-spec.org/latest/), another project of ours, installed).

Conceptually, we decided to separate the loading and deserialization logic to spare you from needing to rewrite your logic when you move different model types to different storage locations.
This way, you need to only define one loader per storage location and one artifact class per artifact type instead of the product of the two. 
Speaking of artifacts, let's continue with the artifact class.

## Using the `Artifact` class
The main purpose of the `Artifact` class is to enable you to load (deserialize) artifacts in a type-safe way which also enables all sorts of autocompletion and security features in your IDE, thereby improving your developer experience.
When implementing the custom `Artifact` you have to override the deserialization method, `deserialize` which needs to assign the deserialized object(s) to the `self._value` property..
You can use the `self.path` attribute to access the local filepath to the serialized artifact. This is provided by the `.load()` method of the appropriate `ArtifactLoader` that you have to pass upon instantiation.
The artifact then exposes the wrapped object with the `.value` attribute, which returns the value of the internal `self._value` property.
Loading an artifact is lazy. 
If you want to deserialize the artifact at a specific point, you can do so via the implemented `deserialize()` method.
Otherwise, `deserialize()` is called internally when you first access the value. 
To provide a minimal example, here is how you could implement an `Artifact` for loading `numpy` arrays. 

```python
import numpy as np
from nnbench.types import Artifact
from loaders import LocalArtifactLoader

class NumpyArtifact(Artifact):
    def deserialize(self) -> None:
        self._value = np.load(self.path)

array_artifact = NumpyArtifact(LocalArtifactLoader('path/to/array'))
print(array_artifact.value)
```
## Using the `ArtifactCollection`
The `ArtifactCollection` does not need you to override any base class methods.
Instead it is a convenience wrapper around a `list` to enable you to iterate over the artifacts or their values using `ArtifactCollection.values()`.
You can instantiate an `ArtifactCollection` by passing a single artifact or an iterable containing some upon instantiation.
Then you can add more with the `add` method.
