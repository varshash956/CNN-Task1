"""
Task 1: CNN image classification using pretrained ResNet-18.
Run:
    python cnn_inference.py --download-samples
    python cnn_inference.py
"""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import sys

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_images"

# Public example images hosted by Wikimedia Commons.
SAMPLE_URLS = {
    "dog.jpg": "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    "eagle.jpg": "https://upload.wikimedia.org/wikipedia/commons/1/1c/Bald_Eagle_Portrait.jpg",
    "tiger.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/56/Tiger.50.jpg",
    "butterfly.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Monarch_In_May.jpg",
}


def download_samples():
    SAMPLE_DIR.mkdir(exist_ok=True)
    for filename, url in SAMPLE_URLS.items():
        destination = SAMPLE_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already downloaded: {filename}")
            continue
        print(f"Downloading {filename} ...")
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read()
            if not data:
                raise ValueError("The downloaded file is empty.")
            destination.write_bytes(data)
            # Confirm that the downloaded file is a readable image.
            with Image.open(destination) as image:
                image.verify()
            print(f"Saved: {destination}")
        except Exception as error:
            if destination.exists():
                destination.unlink()
            print(f"Could not download {filename}: {error}")
            print("You can place any JPG/PNG image with this filename in sample_images/.")
    print("\nSample download step finished.")


def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    return model, preprocess, labels


def predict_five_images():
    image_paths = sorted(
        path for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if SAMPLE_DIR.exists() else []

    if len(image_paths) < 5:
        print("Please keep at least five JPG/PNG images in the sample_images folder.")
        print("Run: python cnn_inference.py --download-samples")
        sys.exit(1)

    image_paths = image_paths[:5]
    model, preprocess, labels = load_model()

    print("\nPRETRAINED CNN: RESNET-18")
    print("=" * 55)
    with torch.inference_mode():
        for index, path in enumerate(image_paths, start=1):
            try:
                image = Image.open(path).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0)
                scores = model(input_tensor)[0]
                probabilities = torch.nn.functional.softmax(scores, dim=0)
                top_probabilities, top_indices = torch.topk(probabilities, 3)

                print(f"\nInput {index}: {path.name}")
                for rank, (probability, class_index) in enumerate(
                    zip(top_probabilities.tolist(), top_indices.tolist()), start=1
                ):
                    print(
                        f"  {rank}. {labels[class_index]} "
                        f"— confidence: {probability * 100:.2f}%"
                    )
            except Exception as error:
                print(f"\nCould not process {path.name}: {error}")

    print("\nNote: Confidence scores are model estimates, not guarantees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN inference on five sample images.")
    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Download five example images into sample_images/."
    )
    args = parser.parse_args()

    if args.download_samples:
        download_samples()
    else:
        predict_five_images()
