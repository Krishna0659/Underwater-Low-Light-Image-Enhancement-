import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class UnderwaterPairedDataset(Dataset):
    """
    Dataset loader for paired underwater images (inputs and ground-truth targets).
    Supports optional RAM preloading to eliminate disk I/O latency.
    """
    def __init__(
        self,
        input_dir,
        target_dir,
        file_list=None,
        is_train=True,
        img_size=(256, 256),
        preload_to_ram=True
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.is_train = is_train
        self.img_size = img_size
        self.preload_to_ram = preload_to_ram

        if file_list is not None:
            self.filenames = file_list
        else:
            in_files = set(os.listdir(input_dir))
            tgt_files = set(os.listdir(target_dir))
            common = sorted(list(in_files.intersection(tgt_files)))
            self.filenames = [f for f in common if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        self.cached_inputs = None
        self.cached_targets = None

        if self.preload_to_ram:
            print(f"Preloading {len(self.filenames)} {'train' if is_train else 'val'} image pairs into RAM...", flush=True)
            self.cached_inputs = np.zeros((len(self.filenames), 3, img_size[0], img_size[1]), dtype=np.float32)
            self.cached_targets = np.zeros((len(self.filenames), 3, img_size[0], img_size[1]), dtype=np.float32)

            for i, fname in enumerate(self.filenames):
                inp_p = os.path.join(input_dir, fname)
                tgt_p = os.path.join(target_dir, fname)

                im_in = Image.open(inp_p).convert('RGB')
                im_tgt = Image.open(tgt_p).convert('RGB')

                if im_in.size != img_size:
                    im_in = im_in.resize(img_size, Image.BILINEAR)
                    im_tgt = im_tgt.resize(img_size, Image.BILINEAR)

                self.cached_inputs[i] = np.array(im_in, dtype=np.float32).transpose(2, 0, 1) / 255.0
                self.cached_targets[i] = np.array(im_tgt, dtype=np.float32).transpose(2, 0, 1) / 255.0

            print(f"Preloaded into RAM successfully!", flush=True)

    def __len__(self):
        return len(self.filenames)

    def _augment(self, inp_tensor, tgt_tensor):
        if random.random() > 0.5:
            inp_tensor = torch.flip(inp_tensor, dims=[2])
            tgt_tensor = torch.flip(tgt_tensor, dims=[2])

        if random.random() > 0.5:
            inp_tensor = torch.flip(inp_tensor, dims=[1])
            tgt_tensor = torch.flip(tgt_tensor, dims=[1])

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            inp_tensor = torch.rot90(inp_tensor, k, [1, 2])
            tgt_tensor = torch.rot90(tgt_tensor, k, [1, 2])

        return inp_tensor, tgt_tensor

    def __getitem__(self, idx):
        if self.preload_to_ram and self.cached_inputs is not None:
            inp_tensor = torch.from_numpy(self.cached_inputs[idx])
            tgt_tensor = torch.from_numpy(self.cached_targets[idx])
        else:
            fname = self.filenames[idx]
            inp_path = os.path.join(self.input_dir, fname)
            tgt_path = os.path.join(self.target_dir, fname)

            inp_img = Image.open(inp_path).convert('RGB').resize(self.img_size, Image.BILINEAR)
            tgt_img = Image.open(tgt_path).convert('RGB').resize(self.img_size, Image.BILINEAR)

            inp_tensor = TF.to_tensor(inp_img)
            tgt_tensor = TF.to_tensor(tgt_img)

        if self.is_train:
            inp_tensor, tgt_tensor = self._augment(inp_tensor, tgt_tensor)

        return {
            'input': inp_tensor,
            'target': tgt_tensor,
            'filename': self.filenames[idx]
        }


def get_train_val_loaders(
    input_dir,
    target_dir,
    val_ratio=0.1,
    batch_size=16,
    num_workers=0,
    seed=42,
    img_size=(256, 256),
    preload_to_ram=True
):
    in_files = set(os.listdir(input_dir))
    tgt_files = set(os.listdir(target_dir))
    common = sorted(list(in_files.intersection(tgt_files)))
    all_files = [f for f in common if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    random.seed(seed)
    shuffled = list(all_files)
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_files = sorted(shuffled[:val_count])
    train_files = sorted(shuffled[val_count:])

    print(f"Dataset split: {len(train_files)} training samples, {len(val_files)} validation samples.")

    train_dataset = UnderwaterPairedDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        file_list=train_files,
        is_train=True,
        img_size=img_size,
        preload_to_ram=preload_to_ram
    )

    val_dataset = UnderwaterPairedDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        file_list=val_files,
        is_train=False,
        img_size=img_size,
        preload_to_ram=preload_to_ram
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
