# An example on benchmarking models with ZenML

This example contains the MNIST example as a ZenML pipeline, including an evaluation step that collects and runs a benchmark suite on the newly trained model.
It logs the results to the step directly as metadata, where the data scientist can immediately inspect them in the ZenML dashboard, or process them in scripts with the client's model metadata APIs.
