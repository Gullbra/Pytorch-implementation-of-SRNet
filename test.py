"""This module is used to test the Srnet model."""

print("Importig modules for SRNet test...")

from glob import glob
from pathlib import Path
import torch
import imageio as io

if __name__ == "__main__":
  from model import Srnet
else:
  from .model import Srnet

print("Modules imported.")


def test_srnet(
  cover_img_dir: Path, 
  stego_img_dir: Path,
  checkpoint_path: Path = Path("./srnet_model/checkpoints/Srnet_model_weights.pt"), 
  test_batch_size: int = 40,
  verbose: bool = False
) -> float:
  
  # Loading images
  if verbose:
    print("Loading images...")
  if not cover_img_dir.exists():
    raise FileNotFoundError("Cover directory does not exist.")
  if not stego_img_dir.exists():
    raise FileNotFoundError("Stego directory does not exist.")

  cover_image_paths = sorted(cover_img_dir.glob("*.png"))
  stego_image_paths = sorted(stego_img_dir.glob("*.png"))
  
  if not cover_image_paths:
    raise FileNotFoundError(f"No .png files found in {cover_img_dir.resolve()}.")
  if not stego_image_paths:
    raise FileNotFoundError(f"No .png files found in {stego_img_dir.resolve()}.")

  min_images = min(len(cover_image_paths), len(stego_image_paths))
  if verbose:
    print(f"Found {len(cover_image_paths)} cover and {len(stego_image_paths)} stego images")
    print(f"Using {min_images} pairs for testing")
    print()

  # Initialize model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if verbose:
    print(f"Using device: {device}")

  model = Srnet().to(device)
  model.eval()  # Set to evaluation mode

  # Load checkpoint if available
  if checkpoint_path.exists():
    if verbose:
      print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if verbose:
      print("Checkpoint loaded successfully")
  else:
    if verbose:
      print(f"Warning: Checkpoint not found at {checkpoint_path}")
      print("Using randomly initialized weights (model is untrained)")

  test_accuracy = []
  
  # Disable gradient computation for inference
  # Since aren't updating weights, this saves memory and computations
  with torch.no_grad():  
    for idx in range(0, min_images, test_batch_size // 2):
      # Get batches (handle last batch being smaller)
      batch_size = min(test_batch_size // 2, min_images - idx)
      cover_batch = cover_image_paths[idx : idx + batch_size]
      stego_batch = stego_image_paths[idx : idx + batch_size]
      
      # Build interleaved batch more efficiently
      batch_paths = []
      batch_labels = []
      for cover_path, stego_path in zip(cover_batch, stego_batch):
        batch_paths.extend([stego_path, cover_path])
        batch_labels.extend([1, 0])
      
      # Load images directly to device
      image_list = []
      for img_path in batch_paths:
        img = io.imread(img_path)
        img_tensor = torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0)
        image_list.append(img_tensor)
      
      # Stack into batch tensor
      image_tensor = torch.stack(image_list)
      batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
      
      # Forward pass
      outputs = model(image_tensor)
      predictions = outputs.argmax(dim=1)
      
      # Calculate accuracy
      accuracy = (predictions == batch_labels_tensor).float().mean() * 100.0
      test_accuracy.append(accuracy.item())
      
      # Progress indicator
      print(f"Processed batch {idx // (test_batch_size // 2) + 1}, Accuracy: {accuracy:.2f}%")
  
  avg_accuracy = sum(test_accuracy) / len(test_accuracy)
  print(f"\nFinal test accuracy = {avg_accuracy:.2f}%")
  
  return avg_accuracy


if __name__ == "__main__":
  cover_dir = Path("../images/BOSSbase_1.01[.png]/test/cover")
  stego_dir = Path("../images/BOSSbase_1.01[.png]/test/stego")

  if not cover_dir.exists():
    raise FileNotFoundError("Cover directory does not exist.")
  
  if not stego_dir.exists():
    raise FileNotFoundError("Stego directory does not exist.")

  test_srnet(
    cover_img_dir = cover_dir,
    stego_img_dir = stego_dir,
    verbose=True
  )



  # # pylint: disable=E1101
  # images = torch.empty((test_batch_size, 1, 256, 256), dtype=torch.float)
  # # pylint: enable=E1101
  # test_accuracy = []

  # for idx in range(0, len(cover_image_paths), test_batch_size // 2):
  #   cover_batch = cover_image_paths[idx : idx + test_batch_size // 2]
  #   stego_batch = stego_image_paths[idx : idx + test_batch_size // 2]

  #   batch = []
  #   batch_labels = []

  #   xi = 0
  #   yi = 0
  #   for i in range(2 * len(cover_batch)):
  #     if i % 2 == 0:
  #       batch.append(stego_batch[xi])
  #       batch_labels.append(1)
  #       xi += 1
  #     else:
  #       batch.append(cover_batch[yi])
  #       batch_labels.append(0)
  #       yi += 1
  #   # pylint: disable=E1101
  #   for i in range(test_batch_size):
  #     images[i, 0, :, :] = torch.tensor(io.imread(batch[i])).cuda()
  #   image_tensor = images.cuda()
  #   batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
  #   # pylint: enable=E1101
  #   outputs = model(image_tensor)
  #   prediction = outputs.data.max(1)[1]

  #   accuracy = (
  #     prediction.eq(batch_labels.data).sum()
  #     * 100.0
  #     / (batch_labels.size()[0])
  #   )
  #   test_accuracy.append(accuracy.item())

  # print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):%.2f}")
