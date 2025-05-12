"""
The GPT-2 Language Model implemented in MLX, for Apple Silicon.
Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import logging
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Log INFO level messages in console
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


DATA_TYPE = mx.float16
CHAR_SIZE = 4544  # vocab size for a character-level LLM, padded to closest multiple of 64 for performance
EMBD_SIZE = 256  # length of an embedding vector
CTXT_SIZE = 256  # length of context window
ATTN_HEAD = 4  # number of attention heads
HEAD_SIZE = EMBD_SIZE // ATTN_HEAD
DROP_RATE = 0.0  # portion of neurons to forget randomly during training
NUM_BLOCKS = 4  # number of transformer blocks


class CausalSelfAttention(nn.Module):
    """
    Muti-head causal self attention without bias, implemented in MLX.
    """

    def __init__(self):
        super().__init__()

        # Query, Key, Value projections for all attention heads in a single layer.
        self.qkv = nn.Linear(EMBD_SIZE, 3 * EMBD_SIZE, bias=False)

        # Output projection
        self.output_linear = nn.Linear(EMBD_SIZE, EMBD_SIZE, bias=False)

        # Dropout layers
        self.attention_dropout = nn.Dropout(DROP_RATE)
        self.output_dropout = nn.Dropout(DROP_RATE)

        self.__scale_factor = 1.0 / math.sqrt(HEAD_SIZE)
        self.__causal_mask = mx.tril(mx.ones((CTXT_SIZE, CTXT_SIZE), dtype=DATA_TYPE))

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape

        # Calculate query, key, values for all heads in batch then split the results
        Q, K, V = self.qkv(x).split(3, axis=2)

        # Ensure Q, K, V are in shape (B, ATTN_HEAD, T, HEAD_SIZE)
        Q = Q.reshape(B, T, ATTN_HEAD, HEAD_SIZE).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, ATTN_HEAD, HEAD_SIZE).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, ATTN_HEAD, HEAD_SIZE).transpose(0, 2, 1, 3)

        # Matrix multiply Query and Key
        # Then scale it by a factor to prevent extreme values from dominating in softmax
        attentions = (Q @ K.transpose(0, 1, 3, 2)) * self.__scale_factor

        # Apply causal masking
        attentions = mx.where(self.__causal_mask[:T, :T] == 0, -1e9, attentions)
        attentions = mx.softmax(attentions, axis=-1)
        attentions = self.attention_dropout(attentions)

        y = attentions @ V
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.output_linear(y)
        y = self.output_dropout(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron without bias, implemented in MLX.
    """

    def __init__(self):
        super().__init__()
        self.input_linear = nn.Linear(EMBD_SIZE, 4 * EMBD_SIZE, bias=False)
        self.output_linear = nn.Linear(4 * EMBD_SIZE, EMBD_SIZE, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(DROP_RATE)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.input_linear(x)
        x = self.gelu(x)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block without bias, implemented in MLX.
    """

    def __init__(self):
        super().__init__()
        self.csa = CausalSelfAttention()
        self.mlp = MLP()
        self.layer_norm_1 = nn.LayerNorm(dims=EMBD_SIZE, bias=False)
        self.layer_norm_2 = nn.LayerNorm(dims=EMBD_SIZE, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.csa(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class GPT(nn.Module):
    """
    The GPT model implemented in MLX.
    """

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(CHAR_SIZE, EMBD_SIZE)  # token embeddings
        self.wpe = nn.Embedding(CTXT_SIZE, EMBD_SIZE)  # positional embeddings
        self.transformer_blocks = nn.Sequential(
            *[Block() for _ in range(NUM_BLOCKS)],
        )
        self.final_layer_norm = nn.LayerNorm(dims=EMBD_SIZE, bias=False)  # final layernorm
        self.final_linear = nn.Linear(EMBD_SIZE, CHAR_SIZE, bias=False)  # output projection

        self.__init_weights()
        self.__total_params = round(sum(v.size for _, v in tree_flatten(self.parameters())) / 1e6, 2)
        LOGGER.info(f"Model size: {self.__total_params}M parameters")

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(T, dtype=mx.uint16))
        x = tok_emb + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear(x)
        return logits

    def __init_weights(self):
        normal_init = nn.init.normal(mean=0.0, std=0.02, dtype=DATA_TYPE)
        residual_init = nn.init.normal(mean=0.0, std=(0.02 / math.sqrt(2 * NUM_BLOCKS)), dtype=DATA_TYPE)

        new_weights = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                assert not hasattr(module, "bias"), f"Module {name} should not have bias"
                if "output_linear" in name:
                    new_weights.append((name + ".weight", residual_init(module.weight)))
                else:
                    new_weights.append((name + ".weight", normal_init(module.weight)))
            elif isinstance(module, nn.Embedding):
                new_weights.append((name + ".weight", normal_init(module.weight)))

            # Ensure everything is in float16
            module.set_dtype(DATA_TYPE)

        self = self.update(tree_unflatten(new_weights))

    def inference(
        self,
        context: mx.array,
        limit: int,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Generate new character IDs based with the given context and settings.
        """

        B, T = context.shape
        assert B == 1, "Batch size must be one at inference time!"
        assert T <= CTXT_SIZE, "Context window is too long at inference time!"
        assert limit <= CTXT_SIZE, f"Generation limit must be <= {CTXT_SIZE}"

        for _ in range(limit):
            context = context[:, -CTXT_SIZE:]  # crop context
            logits = self(context)  # forward pass
            logits = logits[:, -1, :]  # get logits for the next token
            logits = logits / temperature  # apply desired temperature
            new_char_id = mx.random.categorical(logits, num_samples=1)
            context = mx.concatenate((context, new_char_id), axis=1)

        # Only return the newly generated character IDs
        return context[0, -limit:]
