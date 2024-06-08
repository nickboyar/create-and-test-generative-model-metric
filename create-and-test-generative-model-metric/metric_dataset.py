import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import effects

class MetricDataset(Dataset):
    def __init__(self, path, reshape_to, generated, sample_size, effect, power):
        self.path = path
        self.reshape_to = reshape_to
        self.generated = generated
        self.sample_size = sample_size
        self.effect = effect
        self.power = power
        self.img_path_list = self._get_image_list()
        
    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["np", "png", "jpg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        image_list.sort()
        return image_list[:self.sample_size]

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        return im.resize((size, size), resample=Image.BICUBIC)

    def read_image_to_np(self, path, size, generated):
        im = Image.open(path)
        im = self._center_crop_and_resize(im, size)
        if generated:
            im_np = np.array(im)
            if self.effect == 'Noise':
                img = effects.noise(im_np, self.power)
            elif self.effect == 'Sheltering':
                img = effects.sheltering(im_np, self.power)
            elif self.effect == 'Exchange':
                img = effects.exchange(im_np, self.power)        
            else:
                img = im_np
            return img.astype(np.float32)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        x = self.read_image_to_np(img_path, self.reshape_to, self.generated)
        return x