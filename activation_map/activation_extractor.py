from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

import torch
from torch import nn


class ActivationExtractor:
    def __init__(
        self,
        model: nn.Module,
        filter_fn: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        r"""
        Initializes the ActivationExtractor with a model and an optional filter function.

        Args:
            model (nn.Module): The PyTorch model to hook into for activation extraction.
            filter_fn (Optional[Callable[[str, nn.Module], bool]]): A function that takes a layer's name and the layer
                itself, and returns a boolean indicating whether to hook that layer. If None, a default filter is used
                that captures only `nn.Conv2d` layers.
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self.filter_fn = filter_fn if filter_fn else self.default_filter_fn

    def __del__(self):
        """
        Ensures that hooks are cleared when the object is deleted.
        """
        # self.clear_activations() # Not necessary
        self.clear_hooks()

    def default_filter_fn(self, layer_name: str, layer: nn.Module) -> bool:
        r"""
        Default filter function that determines whether a layer should be hooked.

        Args:
            layer_name (str): The name of the layer.
            layer (nn.Module): The layer module itself.

        Returns:
            bool: True if the layer should be hooked, False otherwise.
        """
        # Example default filter: capture only Conv2d layers
        return isinstance(layer, nn.Conv2d)

    def hook_layer(self, layer_name: str, layer: nn.Module):
        r"""
        Registers a hook on a specific layer to capture its output during the forward pass.

        Args:
            layer_name (str): The name of the layer.
            layer (nn.Module): The layer module to hook.
        """
        if self.filter_fn(layer_name, layer):

            def hook_fn(
                module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
            ):
                r"""
                A hook function that captures the output of a layer during the forward pass.
                And stores it with the capsulated `layer_name` into the `activations` dictionary.

                Args:
                    module (nn.Module): The module to which the hook is attached.
                    input (Tuple[torch.Tensor]): The input tensors to the module.
                    output (torch.Tensor): The output tensor of the module.
                """
                self.activations[layer_name] = output.detach().cpu()

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_hooks(self):
        r"""
        Registers hooks on all layers in the model according to the filter function.
        """
        for name, layer in self.model.named_modules():
            self.hook_layer(name, layer)

    def clear_hooks(self):
        r"""
        Removes all registered hooks from the model, freeing resources.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_activations(self):
        r"""
        Clears all stored activations from previous forward passes.
        """
        self.activations.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        r"""
        Returns the stored activations.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are layer names and values are the corresponding activations.
        """
        return self.activations


# Usage Example
def main():
    pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        # device_map="balanced",
    ).to("cuda")

    # Enable CPU offloading for the model
    pipeline.enable_model_cpu_offload()

    # Custom filter function: only capture activations from `Attention` layers
    def custom_filter_fn(layer_name: str, layer: nn.Module) -> bool:
        from diffusers.models.attention_processor import Attention

        return isinstance(layer, Attention)

    activation_extractor = ActivationExtractor(
        pipeline.unet, filter_fn=custom_filter_fn
    )
    activation_extractor.register_hooks()

    prompt = "A futuristic cityscape at sunset"
    output: OrderedDict = pipeline(prompt, num_inference_steps=1)  # type: ignore
    images: List[PIL.Image.Image] = output["images"]
    image = images[0]
    # raise NotImplementedError()
    image.show()

    # Accessing filtered activation maps
    activations = activation_extractor.get_activations()
    for layer_name, activation in activations.items():
        print(f"Layer: {layer_name}, Activation shape: {activation.shape}")
        # Visualize activation map
        activation_map = activation[0].cpu().numpy()

    # Clean up hooks after use
    activation_extractor.clear_hooks()


if __name__ == "__main__":
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    import cv2
    import matplotlib.pyplot as plt

    import numpy as np
    import PIL
    import PIL.Image

    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg", "pdf")

    from diffusers import StableDiffusionPipeline

    try:
        main()
    except Exception as e:
        import bdb, code, pdb, sys, traceback

        if sys.gettrace() is not None:
            # if already in debugging mode, raise exception
            raise e

        te, val, tb = sys.exc_info()
        if isinstance(te, type) and (
            issubclass(te, bdb.BdbQuit)
            or issubclass(te, KeyboardInterrupt)
            or issubclass(te, SystemExit)
        ):
            # exit if explict BdbQuit() was raised
            # as of exiting inner debugging
            exit()
        traceback.print_exc()
        print(f"{te=}, {val=}")
        pdb.post_mortem()
        # pdb.post_mortem(tb)
