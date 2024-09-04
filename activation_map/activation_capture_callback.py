import os
import numpy as np
import torch
from torch import nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.callbacks import PipelineCallback
from typing import Any, Dict, Callable, Optional, List, Union
from activation_extractor import ActivationExtractor


class ActivationCaptureCallback(PipelineCallback):
    tensor_inputs: List[str] = ["latents"]  # type: ignore

    def __init__(
        self,
        model: nn.Module,
        filter_fn: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        """
        Initializes the callback with an ActivationExtractor and an empty list to store activations per timestep.

        Args:
            model (nn.Module): The PyTorch model whose activations are to be extracted.
            filter_fn (Optional[Callable[[str, nn.Module], bool]]): A function to filter which layers should have activations captured.
        """
        super().__init__()
        self.activations_per_timestep: Dict[int, Dict[str, torch.Tensor]] = {}
        self.activation_extractor = ActivationExtractor(model, filter_fn)
        self.activation_extractor.register_hooks()

    def callback_fn(
        self,
        pipeline: DiffusionPipeline,
        step_index: int,
        timesteps: torch.IntTensor,
        callback_kwargs: Dict[str, Any],
    ):
        """
        Captures the model's activations at each timestep.

        Args:
            pipeline (DiffusionPipeline): The pipeline instance.
            step_index (int): The current step index in the denoising loop.
            timestep (int): The current timestep in the denoising loop.
            callback_kwargs: (Dict[str, Any]) containing tensors like 'latents'.
        """
        activations = self.activation_extractor.get_activations().copy()
        activations["z_latents_output"] = callback_kwargs["latents"]
        self.activations_per_timestep[int(timesteps.item())] = activations
        self.activation_extractor.clear_activations()

        return callback_kwargs

    def get_activations(
        self, timesteps: Optional[Union[int, List[int]]] = None
    ) -> Union[Dict[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Returns the activations captured by the callback.

        Args:
            timesteps (Optional[Union[int, List[int]]]): The specific timestep(s) for which to get activations. If None, all activations are returned.

        Returns:
            Union[OrderedDict[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor], None]:
            - If no timestep is specified, returns an OrderedDict of all activations.
            - If a single timestep is specified, returns the activations for that timestep.
            - If a list of timesteps is specified, returns a dictionary of activations for those timesteps.
            - Raises a ValueError if no activations were captured.
        """
        if timesteps is None:
            return self.activations_per_timestep
        try:
            if isinstance(timesteps, int):
                return self.activations_per_timestep[timesteps]
            if isinstance(timesteps, list):
                return {
                    timestep: self.activations_per_timestep[timestep]
                    for timestep in timesteps
                    if timestep in self.activations_per_timestep
                }
        except KeyError:
            raise ValueError("Specified timestep(s) not found in captured activations.")

    def save_activations(self, directory: Union[str, os.PathLike], format: str = "pt"):
        """
        Saves the captured activations to disk.

        Args:
            directory (str | os.PathLike): The directory where the activations should be saved.
            format (str): The format in which to save the activations. Options are 'pt' (PyTorch) or 'npy' (NumPy).
        """
        os.makedirs(directory, exist_ok=True)

        for timestep, activations in self.activations_per_timestep.items():
            timestep_dir = os.path.join(directory, f"timestep_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)

            for layer_name, activation in activations.items():
                layer_name_safe = layer_name.replace(
                    "/", "_"
                )  # Replace '/' to avoid issues in file paths
                if format == "pt":
                    torch.save(
                        activation, os.path.join(timestep_dir, f"{layer_name_safe}.pt")
                    )
                elif format == "npy":
                    np.save(
                        os.path.join(timestep_dir, f"{layer_name_safe}.npy"),
                        activation.cpu().numpy(),
                    )
                else:
                    raise ValueError(f"Unsupported format: {format}")

    def clear_hooks(self):
        """
        Clears hooks from the model after the processing is complete.
        """
        self.activation_extractor.clear_hooks()

    def __del__(self):
        """
        Ensures that hooks are cleared when the object is deleted.
        """
        self.clear_hooks()


# Usage Example
def main():
    pipeline = StableDiffusionPipeline.from_pretrained(
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

    # Instantiate the callback with a model's U-Net
    callback = ActivationCaptureCallback(
        model=pipeline.unet, filter_fn=custom_filter_fn
    )

    # Generate an image with the custom callback
    prompt = "A futuristic cityscape at sunset"
    image = pipeline(
        prompt, callback_on_step_end=callback, num_inference_steps=10
    ).images[0]
    image.show()

    # Access the captured activation maps per timestep
    activations_per_timestep = callback.get_activations()
    for timestep, activations in activations_per_timestep.items():
        if not isinstance(activations, dict):
            raise ValueError(f"No activations were captured on timestep {timestep}.")
        for layer_name, activation in activations.items():
            print(f"  Layer: {layer_name}, Activation shape: {activation.shape}")
        print(f"Total layers in timestep {timestep}: {len(activations.keys())}")
    print(f"Total timesteps: {len(activations_per_timestep)}")

    # Clear the hooks after the operation
    callback.clear_hooks()


if __name__ == "__main__":
    from diffusers.pipelines import StableDiffusionPipeline

    try:
        main()
    except Exception as e:
        import traceback, sys, pdb, bdb, code

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
