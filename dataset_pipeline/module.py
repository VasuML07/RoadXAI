#importations

#used for working with file and folder paths
from pathlib import Path
import random
#used for reading andprocessing images/videos
import cv2
import numpy as np
#used for dl models
import torch
from torch.utils.data import Dataset,DataLoader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RoadDataset(Dataset):

    def __init__(

            self,
            root,
            split = "train",
            #dimensions of input image 512*512 pixels
            image_size = 512,
            #data augmentation will be not applied
            augment = False,
            val_ratio = 0.2,
            seed = 42
    ):

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.samples = self._build_samples(val_ratio, seed)

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.root}")

    def _image_files(self,directory):

        return sorted(

            #searches all folders and subfolders inside directory
            p for p in directory.rglob("*")
            #checks whether file exists and gets file type 
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _find_mask(self,image):

        candidates = [

            #looks for mask in same folder of image
            image.parent / f"{image.stem}.png",
            #goes 1 folder up 
            image.parent.parent / "masks" / f"{image.stem}.png",
            #goes 2 folders up
            image.parent.parent.parent / "masks" / f"{image.stem}.png"

        ]

        for mask in candidates:

            if mask.exists():
                return mask

        return None

    #converts yolo label data to pixel mask
    def _yolo_mask(self,image,label):

        #creates a black mask with same height and width of image
        mask = np.zeros(
            (image.shape[0],image.shape[1]),
            dtype = np.uint8
        )

        if not label.exists():
            return mask

        height, width = mask.shape

        with label.open("r",encoding = "utf-8") as f:

            for line in f:

                #removes extra spaces/newlines
                values = line.strip().split()

                if len(values) < 5:
                    continue

                try:

                    numbers = np.asarray(
                        [float(v) for v in values[1:]],
                        dtype = np.float32
                    )

                except ValueError:
                    continue

                #checks if label contains ploygon coordinates and atleast 2 points are required each point has 2 values x and y
                if len(numbers) >=  6 and len(numbers) % 2 == 0:

                    #converts numbers to (x,y) pairs
                    points = numbers.reshape(-1,2)

                    points[:, 0] *= width
                    points[:,1] *= height

                    points = np.round(points).astype(np.int32)
                    #draws polygon on mask with white color
                    cv2.fillPoly(mask, [points], 255)

                #if only 4 exists treats it as bounding box 
                elif len(numbers) == 4:

                    x_center, y_center, box_width, box_height = numbers

                    #calculates left edge 
                    x1 = int((x_center - box_width / 2) * width)
                    #calculates top edge
                    y1 = int((y_center - box_height / 2) * height)
                    #calculates right edge
                    x2 = int((x_center + box_width / 2) * width)
                    #calculates bottom edge
                    y2 = int((y_center + box_height / 2) * height)

                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))

                    #draws a filed white rectangle on mask 
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)


        return mask


    def _load_sample(self,image):

        mask = self._find_mask(image)

        if mask is not None:
            return image, mask

        #if no mask is found looks for yolo label 
        label = image.parent.parent / "labels" / f"{image.stem}.txt"

        if label.exists():
            return image, label

        return None

    #collects all valid image_label/mask pairs form dataset 
    def _collect(self):
        samples = []

        for image in self._image_files(self.root):
            sample = self._load_sample(image)

            if sample is not None:
                samples.append(sample)

        return samples

    #builds final train/test split list 
    def _build_samples(self, val_ratio, seed):

        #creates path to split dataset 
        split_dir = self.root / self.split

        if split_dir.exists():

            samples = self._collect_from_split(split_dir)

        else:

            samples = self._collect()

            if len(samples) < 2:
                return samples

            rng = random.Random(seed)
            rng.shuffle(samples)

            #calculates hown many smaples should go for validation 
            val_count = max(1, int(len(samples) * val_ratio))

            if self.split == "val":
                samples = samples[:val_count]
            elif self.split == "train":
                samples = samples[val_count:]

        return samples


    #collects valid image mask/label pairs for train/test split folder
    def _collect_from_split(self, split_dir):

        images_dir = split_dir / "images"

        if images_dir.exists():
            image_files = self._image_files(images_dir)
        else:
            image_files = self._image_files(split_dir)

        samples = []

        for image in image_files:
            mask = self._find_mask(image)

            if mask is not None:
                samples.append((image, mask))
                continue

            label = split_dir / "labels" / f"{image.stem}.txt"

            if label.exists():
                samples.append((image, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, annotation = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        #opencb reads images as bgr converts to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if annotation.suffix.lower() == ".txt":
            mask = self._yolo_mask(image, annotation)
        else:
            mask = cv2.imread(str(annotation), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise RuntimeError(f"Unable to read mask: {annotation}")

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        #does vertical and horizontal flip augmenttation 
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if self.augment and random.random() < 0.2:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        #normalizes images tp (0,1)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        #converts mask to binary  so 0 = background 1 = object 
        image = torch.from_numpy(image).permute(2, 0, 1)
        #converts image to pytorch tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def create_dataloaders(

        dataset_root,
        image_size=512,
        #model recieves 8 images at once 
        batch_size=8,
        num_workers=0,
        val_ratio=0.2,
        seed=42
    ):

    train_dataset = RoadDataset(
        dataset_root,
        split="train",
        image_size=image_size,
        #enables random image augmentation
        augment=True,
        val_ratio=val_ratio,
        seed=seed
    )

    val_dataset = RoadDataset(
        dataset_root,
        split="val",
        image_size=image_size,
        augment=False,
        val_ratio=val_ratio,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #randomly shiffles training samples every epoch 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        #keeps last batch smaller 
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader