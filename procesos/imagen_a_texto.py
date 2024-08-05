from collections.abc import Callable

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image


class Imagen_a_texto:
    pipe: ResNetForImageClassification | None = None

    def carga_modelo(self) -> None:
        pipe = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.pipe = pipe

    def clasificar(self,img:Image.Image) -> str:
        if not self.pipe:
            raise RuntimeError("El Pipeline no ha sido cargado")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        inputs = processor(img, return_tensors="pt")

        with torch.no_grad():
            logits = self.pipe(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        return self.pipe.config.id2label[predicted_label]