import io
import os
from typing import Optional

import requests
import torch
from huggingface_hub import get_token

try:
    # Prefer the dedicated FLUX.2 pipeline when available.
    from diffusers import Flux2Pipeline as _FluxPipeline
except ImportError:  # pragma: no cover - depends on diffusers version
    from diffusers import FluxPipeline as _FluxPipeline

REMOTE_TEXT_ENCODER_URL = os.getenv(
    "FLUX_REMOTE_TEXT_ENCODER_URL",
    "https://remote-text-encoder-flux-2.huggingface.co/predict",
)


def _resolve_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token missing. Set HF_TOKEN or run `huggingface-cli login` "
            "to access FLUX.2-dev and its remote text encoder."
        )
    return token


class Flux2Generator:
    def __init__(self, model_name: str = "black-forest-labs/FLUX.2-dev", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.token = _resolve_hf_token()

        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        print(f"Loading {model_name} with {_FluxPipeline.__name__} on {device}...")

        # Token is passed explicitly so private model access works even without cached login.
        self.pipe = _FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=self.token,
        ).to(device)

    def remote_text_encoder(self, prompt: str) -> torch.Tensor:
        response = requests.post(
            REMOTE_TEXT_ENCODER_URL,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get embeddings ({response.status_code}): {response.text}"
            )

        prompt_embeds = torch.load(io.BytesIO(response.content), map_location=self.device)
        return prompt_embeds.to(self.device)

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        image: Optional[str] = None,
    ):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Getting embeddings from remote encoder...")
        prompt_embeds = self.remote_text_encoder(prompt)

        print("Generating image...")
        output = self.pipe(
            prompt_embeds=prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        ).images[0]

        return output
