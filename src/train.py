import logging
import math
import pickle
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map
from tqdm import tqdm

from models.gpt2 import (
    CTXT_SIZE,  # length of context window
    GPT,
)
from utils.path import DATA_FOLDER, MODELS_FOLDER

# Log INFO level messages in console
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

TRAIN_META = DATA_FOLDER / "training_metadata.pkl"
TRAIN_DATA = DATA_FOLDER / "train.bin"
VALIDATION_DATA = DATA_FOLDER / "val.bin"

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 3


def count_batches(filepath: Path) -> int:
    """Calculate the total number of batches for the given data file."""
    data = np.memmap(filepath, dtype=np.uint16, mode="r")
    return math.ceil((len(data) - CTXT_SIZE) / BATCH_SIZE)


def get_batch(filepath: Path):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(filepath, dtype=np.uint16, mode="r")
    data = mx.array(data, dtype=mx.uint16)

    assert len(data) > CTXT_SIZE, "Data is too short for the given content window length"
    indices = np.arange(len(data) - CTXT_SIZE)
    np.random.shuffle(indices)

    # Yield batches
    for i in range(0, len(indices), BATCH_SIZE):
        # Get batch indices (handle last batch that might be smaller)
        batch_indices = indices[i : i + BATCH_SIZE]

        x_batch = []
        y_batch = []

        # Create input-target pairs
        for idx in batch_indices:
            idx = int(idx)

            # Input: sequence starting at index idx with length CTXT_SIZE
            x = data[idx : idx + CTXT_SIZE]

            # Target: sequence starting at index idx+1 with length CTXT_SIZE
            y = data[idx + 1 : idx + CTXT_SIZE + 1]

            x_batch.append(x)
            y_batch.append(y)

        # Stack the batches into tensors
        x_batch = mx.stack(x_batch)
        y_batch = mx.stack(y_batch)
        yield x_batch, y_batch


def loss_fn(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    # Cast logits to float32 for numerical stability
    logits = model(x).astype(mx.float32)
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    y = y.reshape(B * T)
    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    return loss


def main():
    model = GPT()
    mx.eval(model.parameters())
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, eps=1e-4)

    # Calculate total batches for progress bars
    train_batches = count_batches(TRAIN_DATA)
    validation_batches = count_batches(VALIDATION_DATA)

    LOGGER.info(f"Begin training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        model.train(True)
        batch_count = 0
        train_loss = 0.0

        # Use tqdm for progress bar
        batch = tqdm(
            get_batch(TRAIN_DATA),
            total=train_batches,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
        )
        for x, y in batch:
            batch_count += 1
            loss, grads = loss_and_grad(model, x, y)

            # Clip gradients to [-1, 1]
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

            optimizer.update(model, grads)
            train_loss += loss.item()
            mx.eval(model.parameters(), optimizer.state)
            batch.set_postfix(loss=f"{(train_loss / batch_count):.4f}")

        model.train(False)  # set eval mode
        batch_count = 0
        validation_loss = 0

        # Use tqdm for progress bar
        batch = tqdm(
            get_batch(VALIDATION_DATA),
            total=validation_batches,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
        )
        for x, y in batch:
            batch_count += 1
            loss = loss_fn(model, x, y)
            validation_loss += loss.item()
            batch.set_postfix(loss=f"{(validation_loss / batch_count):.4f}")

    # Save model
    model_path = MODELS_FOLDER / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    LOGGER.info("Model saved as `model.pkl`.")


if __name__ == "__main__":
    main()
