#importations

from pathlib import Path
import torch

from module import create_dataloaders

DATASETS = [
    "CRACK500",
    "Pothole_Segmentation_YOLOv8",
    "PUBLIC POTHOLE DATASET"
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets"

def verify_dataset(name):
    root = DATASET_ROOT / name

    if not root.exists():
        print(f"{name}: NOT FOUND")
        return False

    try:
        train_loader, val_loader = create_dataloaders(
            dataset_root=root,
            image_size=512,
            #loads 2 images per batch
            batch_size=2,
            num_workers=0
        )

        images, masks = next(iter(train_loader))

        valid = (
            images.ndim == 4
            and masks.ndim == 4
            #checks whetehr image has 3 channels and image size
            and images.shape[1:] == (3, 512, 512)
            #checks mask dimensions 
            and masks.shape[1:] == (1, 512, 512)
            and images.dtype == torch.float32
            and masks.dtype == torch.float32
            #rejects nan and inf values for images
            and torch.isfinite(images).all()
            #rejects nan and inf values for masks 
            and torch.isfinite(masks).all()
            and torch.all((masks == 0) | (masks == 1))
        )

        print(
            f"{name}: "
            f"train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, "
            f"image={tuple(images.shape)}, "
            f"mask={tuple(masks.shape)}, "
            f"status={'PASS' if valid else 'FAIL'}"
        )

        return valid


    except Exception as e:
        print(f"{name}: ERROR - {e}")
        return False


def main():
    results = [verify_dataset(name) for name in DATASETS]

    print(
        f"\nDataset verification: "
        f"{sum(results)}/{len(results)} passed"
    )

    if not all(results):
        raise RuntimeError("Dataset verification failed")


if __name__ == "__main__":
    main()
