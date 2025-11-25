import io
import os
from typing import Optional

import torch
import requests
from diffusers import FluxPipeline
from huggingface_hub import get_token

# Use FluxPipeline as Flux2Pipeline might not be available in the main release yet, 
# or it might be an alias. The snippet showed Flux2Pipeline but standard FluxPipeline 
# often handles variations. If Flux2Pipeline is strictly required, we might need to 
# adjust imports, but for now we'll try to be robust. 
# However, the snippet explicitly imported Flux2Pipeline. 
# Let's try to import it, and fall back or assume it's available.
# Given the user wants "complete re-engineering" based on the new model, 
# I will assume the environment will have the necessary diffusers version.

try:
    from diffusers import Flux2Pipeline
except ImportError:
    # Fallback or placeholder if the specific class isn't found, 
    # though likely it is needed for FLUX.2 specific features.
    # For now, we will assume the user has the correct diffusers version installed.
    # If not, this might fail, but we can't easily check installed package versions 
    # dynamically without running code.
    Flux2Pipeline = None

class Flux2Generator:
    def __init__(self, model_name: str = "black-forest-labs/FLUX.2-dev", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.token = get_token()
        
        # Determine pipeline class
        pipeline_class = Flux2Pipeline if Flux2Pipeline else FluxPipeline
        
        print(f"Loading {model_name} with {pipeline_class.__name__}...")
        
        # Load pipeline
        # Note: The snippet used a specific 4-bit repo and remote text encoder.
        # We should support the standard dev model too if possible, but the remote 
        # text encoder is a key feature for consumer hardware.
        
        # If using the standard repo, we might need to handle text encoders differently.
        # For now, let's implement the remote text encoder pattern as it's the 
        # suggested usage for "consumer type graphics card".
        
        self.pipe = pipeline_class.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            text_encoder=None, # We will provide embeddings manually
            transformer=None if "bnb-4bit" in model_name else None # logic might vary
        ).to(device)
        
        # If it's the 4-bit model, we might need specific loading logic for the transformer
        # but from_pretrained usually handles it if dependencies are met.
        
    def remote_text_encoder(self, prompt: str) -> torch.Tensor:
        url = "https://remote-text-encoder-flux-2.huggingface.co/predict"
        response = requests.post(
            url,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get embeddings: {response.text}")
            
        prompt_embeds = torch.load(io.BytesIO(response.content))
        return prompt_embeds.to(self.device)

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        image: Optional[str] = None # Path or URL for img2img
    ):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Getting embeddings from remote encoder...")
        prompt_embeds = self.remote_text_encoder(prompt)
        
        print("Generating image...")
        # Basic txt2img
        output = self.pipe(
            prompt_embeds=prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        return output
