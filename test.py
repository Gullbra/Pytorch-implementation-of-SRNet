"""This module is used to test the Srnet model."""

print("Importing modules for SRNet test...")

import argparse
from enum import Enum
from glob import glob
import os
from pathlib import Path
import re
import torch
import random
import imageio.v2 as io

if __name__ == "__main__":
  from model import Srnet
else:
  from .model import Srnet

print("Modules imported.")


class TestMode(Enum):
  BALANCED = 1
  STEGO_ONLY = 2
  COVER_ONLY = 3


def test_srnet(
  cover_img_dir: Path, 
  stego_img_dir: Path,
  checkpoint_path: Path = Path(__file__).parent / "checkpoints", 
  test_batch_size: int = 40,
  verbose: bool = False,
  limit_images: int | None = None,
  mode: TestMode = TestMode.BALANCED
) -> float:
  
  # Loading images
  if verbose:
    print("Loading images...")
    print(f"Checking cover images directory {str(cover_img_dir)}")
  if not cover_img_dir.exists():
    raise FileNotFoundError("Cover directory does not exist.")
  if verbose:
    print(f"Checking stego images directory {str(stego_img_dir)}")
  if not stego_img_dir.exists():
    raise FileNotFoundError("Stego directory does not exist.")

  cover_image_paths = sorted(cover_img_dir.glob("*.png"))
  stego_image_paths = sorted(stego_img_dir.glob("*.png"))
  
  if not cover_image_paths:
    raise FileNotFoundError(f"No .png files found in {cover_img_dir.resolve()}.")
  if not stego_image_paths:
    raise FileNotFoundError(f"No .png files found in {stego_img_dir.resolve()}.")

  # Determine which images to use based on mode
  if mode == TestMode.STEGO_ONLY:
    image_paths = stego_image_paths
    labels = [1] * len(image_paths)  # All stego
    if verbose:
      print(f"Mode: STEGO_ONLY - Testing only on stego images")
  elif mode == TestMode.COVER_ONLY:
    image_paths = cover_image_paths
    labels = [0] * len(image_paths)  # All cover
    if verbose:
      print(f"Mode: COVER_ONLY - Testing only on cover images")
  else:  # BALANCED mode
    min_images = min(len(cover_image_paths), len(stego_image_paths))
    if verbose:
      print(f"Mode: BALANCED - Testing on both cover and stego images")
      print(f"Found {len(cover_image_paths)} cover and {len(stego_image_paths)} stego images")
      print(f"Using {min_images} pairs for testing")

  # Apply limit_images if specified
  if limit_images is not None and limit_images > 0:
    if mode == TestMode.BALANCED:
      if limit_images // 2 < min(len(cover_image_paths), len(stego_image_paths)):
        cover_image_paths = cover_image_paths[:limit_images // 2]
        stego_image_paths = stego_image_paths[:limit_images // 2]
        if verbose:
          print(f"Limiting to {limit_images} images ({limit_images // 2} cover and {limit_images // 2} stego).")
    else:  # STEGO_ONLY or COVER_ONLY
      if limit_images < len(image_paths):
        image_paths = image_paths[:limit_images]
        labels = labels[:limit_images]
        if verbose:
          print(f"Limiting to {limit_images} images.")

  # Initialize model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if verbose:
    print(f"Using device: {device}")

  torch.serialization.add_safe_globals([argparse.Namespace])

  model = Srnet().to(device)
  model.eval()  # Set to evaluation mode

  # Load checkpoint if available
  if checkpoint_path.exists():
    if verbose:
      print(f"Loading checkpoint from: {checkpoint_path}")

    def latest_checkpoint():
      if verbose:
        print(f"Searching for checkpoint(s) in {checkpoint_path}")
      if os.path.exists(checkpoint_path):
          all_chkpts = "".join(os.listdir(checkpoint_path))
          if len(all_chkpts) > 0:
              latest = max(map(int, re.findall(r"\d+", all_chkpts)))
          else:
              latest = None
      else:
          latest = None
      return latest
    
    check_point = latest_checkpoint()

    if not check_point:
      print(f"Checkpoints folder not found in {checkpoint_path}")
    else:
      ckpt = torch.load(checkpoint_path / ("net_" + str(check_point) + ".pt"), map_location=device)
      model.load_state_dict(ckpt["model_state_dict"])
      if verbose:
        print(f"Checkpoint {check_point} loaded successfully")
  else:
    if verbose:
      print(f"Warning: Checkpoint not found at {checkpoint_path}")
      print("Using randomly initialized weights (model is untrained)")
    return

  test_accuracy = []
  
  # Disable gradient computation for inference
  with torch.no_grad():
    if mode == TestMode.BALANCED:
      # Original balanced mode logic
      min_images = min(len(cover_image_paths), len(stego_image_paths))
      for idx in range(0, min_images, test_batch_size // 2):
        # Get batches (handle last batch being smaller)
        batch_size = min(test_batch_size // 2, min_images - idx)
        cover_batch = cover_image_paths[idx : idx + batch_size]
        stego_batch = stego_image_paths[idx : idx + batch_size]
        
        # Build interleaved batch
        batch_paths = []
        batch_labels = []
        for cover_path, stego_path in zip(cover_batch, stego_batch):
          batch_paths.extend([stego_path, cover_path])
          batch_labels.extend([1, 0])
        
        # Check if batch is empty before processing
        if not batch_paths:
          continue
        
        # Load images directly to device
        image_list = []
        for img_path in batch_paths:
          img = io.imread(img_path, pilmode='L')
          img_tensor = torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0)
          image_list.append(img_tensor)
        
        if not image_list:
          print(f"Warning: No images loaded for batch starting at index {idx}")
          continue
        
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
    
    else:  # STEGO_ONLY or COVER_ONLY mode
      num_images = len(image_paths)
      for idx in range(0, num_images, test_batch_size):
        # Get batch (handle last batch being smaller)
        batch_size = min(test_batch_size, num_images - idx)
        batch_paths = image_paths[idx : idx + batch_size]
        batch_labels = labels[idx : idx + batch_size]
        
        if not batch_paths:
          continue
        
        # Load images directly to device
        image_list = []
        for img_path in batch_paths:
          img = io.imread(img_path, pilmode='L')
          img_tensor = torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0)
          image_list.append(img_tensor)
        
        if not image_list:
          print(f"Warning: No images loaded for batch starting at index {idx}")
          continue
        
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
        print(f"Processed batch {idx // test_batch_size + 1}, Accuracy: {accuracy:.2f}%")
  
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

  # Example: Test in balanced mode
  test_srnet(
    cover_img_dir=cover_dir,
    stego_img_dir=stego_dir,
    verbose=True,
    limit_images=200,
    mode=TestMode.BALANCED
  )
  
  # Example: Test only on stego images
  test_srnet(
    cover_img_dir=cover_dir,
    stego_img_dir=stego_dir,
    verbose=True,
    limit_images=100,
    mode=TestMode.STEGO_ONLY
  )
  
  # Example: Test only on cover images
  test_srnet(
    cover_img_dir=cover_dir,
    stego_img_dir=stego_dir,
    verbose=True,
    limit_images=100,
    mode=TestMode.COVER_ONLY
  )