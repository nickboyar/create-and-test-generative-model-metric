import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection
import timm

DINO_MODEL_NAME = 'facebook/dinov2-base'
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
INCEPTION_NAME = 'inception_v3'
INCEPTION_INPUT_SIZE = 299
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embedding_preprocess(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

class DinoEmbedding:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(DEVICE)
        self.input_image_size = self.processor.crop_size["height"]

    @torch.no_grad()
    def get_embedding(self, images):
        images = embedding_preprocess(images, self.input_image_size)
        inputs = self.processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        image_embs = self.model(**inputs)['pooler_output'].cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
    

class ClipEmbedding:
    def __init__(self):
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME).eval().to(DEVICE)
        self.input_image_size = self.processor.crop_size["height"]
        
    @torch.no_grad()
    def get_embedding(self, images):
        images = embedding_preprocess(images, self.input_image_size)
        inputs = self.processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        image_embs = self.model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
    
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class InceptionEmbedding:
    def __init__(self):
        self.model = timm.create_model(INCEPTION_NAME, pretrained=True).eval()
        self.model.fc = Identity()
        self.model = self.model.cuda()
        self.input_image_size = INCEPTION_INPUT_SIZE
        
    @torch.no_grad()
    def get_embedding(self, images):
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        images = torch.nn.functional.interpolate(images, size=(self.input_image_size, 
                                                               self.input_image_size),
                                                               mode="bicubic")
        images = images / 255
        images = images.to(DEVICE)
        image_embs = self.model(images).cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs