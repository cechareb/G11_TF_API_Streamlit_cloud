from collections.abc import Callable

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class Texto_a_Imagen:
    pipe: StableDiffusionPipeline | None = None

    def carga_modelo(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.to(device)
        self.pipe = pipe

    def generar(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_steps: int = 50,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
    ) -> Image.Image:
        if not self.pipe:
            raise RuntimeError("El Pipeline no ha sido cargado")
        return self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=9.0,
            callback=callback,
        ).images[0]


if __name__ == "__main__":
    txt_a_img = Texto_a_Imagen()
    txt_a_img.carga_modelo()

    def callback(step: int, _timestep, _tensor):
        print(f"Paso {step}")

    imagen = txt_a_img.generar(
        "A Renaissance castle in the Loire Valley",
        negative_prompt="low quality, ugly",
        #callback=callback,
    )
    imagen.save("resultado.png")