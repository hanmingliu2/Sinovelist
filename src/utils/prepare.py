"""
Prepare training data for character-level language modeling.
Reference: https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py

Files
-----------
train.bin: contains IDs of characters in training data
val.bin: contains IDs of characters in validation data
meta.pkl: contains the encoder, decoder, and vobaulary size
"""

import logging
import pickle
from typing import List

import numpy as np
import pandas as pd
from path import DATA_FOLDER, NOVELS_FOLDER, NOVELS_METADATA

# Log INFO level messages in console
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def main():
    df = pd.read_csv(NOVELS_METADATA)
    df["filename"] = df["author_name"] + "_" + df["novel_name"] + "_cleaned.txt"
    df["filepath"] = df["filename"].apply(lambda f: NOVELS_FOLDER / f)
    df["is_cleaned"] = df["filepath"].apply(lambda p: p.exists())

    # Check if there is any scraped novel
    df = df[df["is_cleaned"]]
    if df.empty:
        LOGGER.error("Data preparation failed: couldn't find any cleaned novel")
        return

    novels = []
    for filepath in df["filepath"]:
        with open(filepath, "r", encoding="utf-8") as f:
            novels.append(f.read())

    data = "\n\n".join(novels)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    LOGGER.info(f"All the unique characters: {''.join(chars)}")
    LOGGER.info(f"Vocab size: {vocab_size:,}")

    # Create a mapping from characters to integers
    STOI = {ch: i for i, ch in enumerate(chars)}
    ITOS = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> List[int]:
        return [STOI[c] for c in s]

    def decode(ids: List[int]):
        return "".join([ITOS[id] for id in ids])

    # 90:10 train test split
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    LOGGER.info(f"Training set has {len(train_ids):,} tokens")
    LOGGER.info(f"Validation set has {len(val_ids):,} tokens")

    # Convert to numpy arrays
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    # Save as binary files
    train_ids.tofile(DATA_FOLDER / "train.bin")
    val_ids.tofile(DATA_FOLDER / "val.bin")

    # Save the metadata to help us encode/decode later
    metadata = {
        "vocab_size": vocab_size,
        "stoi": STOI,
        "itos": ITOS,
    }
    with open(DATA_FOLDER / "training_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
