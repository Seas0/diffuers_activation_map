# A Simple Activation Map Exporter for `diffuers` pipelines

Draft of a simple activation map exporter for `diffusers` pipelines.

Currently only supports `torch` models and the `unet` module in pipelines.

## TODO List

TODO:

- [ ] Add a simple example
- [ ] Polish the callback and exporter
- [ ] Add support for other models, like these in `transformers`, etc.
      (especially the `CLIP` model for text prompt encoder)
- [ ] Add a proper `ActivationMap` class for managing captured activations
- [ ] Add a hook in the pipeline to recover original image size and restore it for activation maps
