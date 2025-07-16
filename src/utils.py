import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def validate_dataset(csv_path: str, img_dir: str, drop_missing: bool = False) -> pd.DataFrame:
    """Validate dataset files and return the loaded DataFrame.

    Checks that the CSV and image directory exist and contain the expected
    files. Raises informative errors if anything is missing. When
    ``drop_missing`` is ``True``, entries whose images are not found are
    removed from the returned DataFrame instead of raising an error.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.listdir(img_dir):
        raise ValueError(f"Image directory '{img_dir}' is empty")

    df = pd.read_csv(csv_path)
    df['filename'] = df['id_code'].astype(str) + '.png'

    missing = [f for f in df['filename']
               if not os.path.isfile(os.path.join(img_dir, f))]
    if missing:
        if drop_missing:
            df = df[~df['filename'].isin(missing)].reset_index(drop=True)
        else:
            example = ', '.join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} images listed in {csv_path} are missing in "
                f"{img_dir}. Example: {example}")
    return df
