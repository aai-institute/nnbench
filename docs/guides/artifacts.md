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
Additionally, we provide some derived classes out of the box to handle local filepaths using filesystems, which are covered by the fsspec package.
Let us now discuss each class of the framework in detail.

## Using the `ArtifactLoader`
The `ArtifactLoader` is an abstract base class for which you can implement your custom instance by overriding the `load()` method, which needs to return a file path either as string or a path-like object.
You can see an example of it in the `LocalArtifactLoader` implementation that is also provided out of the box by nnbench.
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
The class is then able to handle different filepaths, like S3 and GCS URIs.

## Using the `Artifact` class
The main purpose of the `Artifact` class is to load (deserialize) artifacts in a type-safe way, enabling autocompletion and type inference for your IDE to improve your developer experience.
When implementing a custom `Artifact` subclass, you only have to override the `deserialize` method, which assigns the loaded object(s) to the `Artifact._value` member.
You can use the `self.path` attribute to access the local filepath to the serialized artifact. This is provided by the `.load()` method of the appropriate `ArtifactLoader` that you have to pass upon instantiation.
The artifact then exposes the wrapped object with the `.value` property, which returns the value of the internal `self._value` class member.
Artifacts are loaded lazily.
If you want to deserialize an artifact at a specific point, you can do so by calling the implemented `deserialize()` method.
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
Instead, it is a convenience wrapper around a `list` to enable you to iterate over the artifacts or their values using `ArtifactCollection.values()`.
You can instantiate an `ArtifactCollection` by passing a single artifact or an iterable containing some upon instantiation.
Then you can add more with the `add` method.
To add an artifact to the collection, use the `ArtifactCollection.add()` method.
