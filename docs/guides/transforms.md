# Using transforms to manipulate benchmark records

After a successful benchmark run execution, you end up with your metrics, context, and parameters in a single benchmark record struct.
In general, this data is a best-effort representation of the environment and configuration the benchmarks are run in.

However, in some situations, manual editing and transformation of these records is required.
nnbench exposes the `nnbench.transforms` module to facilitate these transforms.

## Types of transforms: 1->1 vs. N->1 vs. N->N

In nnbench, transforms are grouped by the functional relationship between inputs and outputs.
The easiest case is a 1->1 (one-to-one) transform, which takes a record and produces another.

In the N->1 (N-to-one) case, the transform takes a collection of records and produces a single output record.
This case is very common when computing statistics on records, like mean and variance of target metrics.

In the N->N (N-to-N) case, the transform maps the input record collection to an output collection, generally assumed to be of the same length.
This case is common when mapping records to an equivalent but more easily digestible record format.

The following is an example of a 1->1 transform, which maps the benchmark parameters to representations that are JSON-serializable.

```python
--8<-- "examples/transforms/transforms.py:33:54"
```

In the `MyTransform.apply()` method, the NumPy array is serialized as a list by calling `array.tolist()`, while the model is saved by its checksum only.
In real applications, parametrizing the model with basic Python values will likely take more effort, but this is a first example of how to do it.

The transform is applied on the resulting record, and allows writing the record to JSON without any errors that would normally occur.

```python
--8<-- "examples/transforms/transforms.py:68:75"
```

## Invertible transforms

Borrowing from the same concept in linear algebra, an nnbench `Transform` is said to be **invertible** if there is a function that restores the original record when applied on the transformed record.
For simplicity, the inverse of a transform can be directly defined in the class with the `Transform.iapply()` method.

In general, when designing an invertible transform, it should hold that for any benchmark record `r`, `T.iapply(T.apply(r)) == r`.
A transform is signalled to be invertible if the `Transform.invertible` attribute is set to `True`.

!!! Tip
    While this framework is useful for designing and thinking about transforms, it is not actually enforced by nnbench.
    nnbench will not take any steps to ensure invertibility of transforms, so any transforms should be tested against expected benchmark record data.

## General considerations for writing and using transforms

A few points are useful to keep in mind while writing transforms:

* It is in general not advised to inject arbitrary metadata into records via a transform. If you find yourself needing to supply more metadata, consider using a `ContextProvider` instead.
* When serializing Python values (like the benchmark parameters), be careful to choose a unique representation, otherwise you might not be able to reconstruct model and data versions from written records in a reproducible manner.
* When designing a transform that is not invertible, consider raising a `NotImplementedError` in the `iapply()` method to prevent accidental calls to the (ill-defined) inverse.

## Appendix: The full example code

Here is the full example on how to use transforms for record-to-file serialization:

```python
--8<-- "examples/transforms/transforms.py"
```
