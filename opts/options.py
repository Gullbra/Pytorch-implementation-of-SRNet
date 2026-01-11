"""This module provides method to enter various input to the model training."""
import argparse
from pathlib import Path


def arguments(
  train_dir: Path = Path(__file__ ).parent.parent.parent / "images" / "BOSSbase_1.01[.png]" / "train",
  validation_dir: Path = Path(__file__ ).parent.parent.parent / "images" / "BOSSbase_1.01[.png]" / "val",
  checkpoint_dir: Path = Path(__file__ ).parent.parent / "checkpoints"
) -> str:
  """This function returns arguments."""

  parser = argparse.ArgumentParser()

  for base_path in [(train_dir, ""), (validation_dir, "valid_")]:
    for dir_name in ["cover", "stego"]:
      full_dir = (base_path[0] / dir_name).resolve()

      if not full_dir.exists():
        raise ValueError(f"Directory {full_dir} doesn't exist!")
        
      parser.add_argument(
        f"--{base_path[1]}{dir_name}_path",
        default=str(full_dir)
      )

  parser.add_argument("--checkpoints_dir", default=(str(checkpoint_dir.resolve())))
  parser.add_argument("--num_epochs", type=int, default=50)
  parser.add_argument("--lr", type=float, default=0.001)

  # parser.add_argument("--batch_size", type=int, default=10)
  parser.add_argument("--batch_size", type=int, default=20)

  # parser.add_argument("--train_size", type=int, default=20)
  parser.add_argument("--train_size", type=int, default=400)

  # parser.add_argument("--val_size", type=int, default=10)
  parser.add_argument("--val_size", type=int, default=50)
  

  opt = parser.parse_args()
  return opt
