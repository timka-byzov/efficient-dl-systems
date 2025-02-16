import torch
from torch.utils.data import IterableDataset

class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int,
        batch_size: int,
        drop_long_samples: bool = True,
        pad_token_id: int = 0
    ):
        """
        Args:
            data_path (str): Path to a text file with one training sample per line.
            tokenizer (callable): A function that converts a text string into a list of token IDs.
            max_seq_length (int): Maximum allowed sequence length for a sample.
            batch_size (int): Number of samples per batch.
            drop_long_samples (bool): If True, drop samples longer than max_seq_length;
                                      otherwise, truncate them.
            pad_token_id (int): Token ID used for padding.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.drop_long_samples = drop_long_samples
        self.pad_token_id = pad_token_id

    def __iter__(self):
        batch_token_tensors = []
        batch_attention_masks = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Tokenize and convert to tensor (assumes tokenizer returns a list of ints)
                tokens = self.tokenizer.tokenize(line)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                token_tensor = torch.tensor(token_ids, dtype=torch.long)

                # If the sample is too long, either drop it or truncate it.
                if token_tensor.size(0) > self.max_length:
                    if self.drop_long_samples:
                        continue
                    else:
                        token_tensor = token_tensor[:self.max_length]

                # Build a simple attention mask for the unpadded sample: 1 for valid tokens.
                sample_attention = torch.ones(token_tensor.size(0), dtype=torch.long)

                batch_token_tensors.append(token_tensor)
                batch_attention_masks.append(sample_attention)

                # Once we have enough samples, pad them to a uniform length and yield a batch.
                if len(batch_token_tensors) == self.batch_size:
                    batch_tokens, combined_attention_mask = self.pad_and_build_mask(
                        batch_token_tensors, batch_attention_masks
                    )
                    yield batch_tokens, combined_attention_mask
                    batch_token_tensors = []
                    batch_attention_masks = []

            # Optionally, handle leftover samples (could drop them if incomplete)
            if batch_token_tensors:
                batch_tokens, combined_attention_mask = self.pad_and_build_mask(
                    batch_token_tensors, batch_attention_masks
                )
                yield batch_tokens, combined_attention_mask

    def pad_and_build_mask(self, tokens_list, masks_list):
        """
        Pads a list of token tensors to the maximum length in the batch and constructs a combined attention mask.
        The combined mask is computed per sample as a causal mask (lower triangular) multiplied by the padding mask,
        ensuring tokens can attend only to previous tokens in the same example.
        
        Returns:
            batch_tokens (Tensor): Tensor of shape (batch_size, max_seq_len_in_batch).
            combined_mask (Tensor): Tensor of shape (batch_size, max_seq_len_in_batch, max_seq_len_in_batch).
        """
        # Determine maximum length in the current batch.
        max_len = max(t.size(0) for t in tokens_list)
        padded_tokens = []
        padded_masks = []
        for t, m in zip(tokens_list, masks_list):
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), self.pad_token_id, dtype=t.dtype)])
                m = torch.cat([m, torch.zeros(pad_len, dtype=m.dtype)])
            padded_tokens.append(t)
            padded_masks.append(m)

        # Stack into batch tensors.
        batch_tokens = torch.stack(padded_tokens)      # shape: (batch_size, max_len)
        batch_masks = torch.stack(padded_masks)          # shape: (batch_size, max_len)

        # For each sample, create a causal mask (lower triangular matrix).
        causal_masks = []
        for i in range(batch_tokens.size(0)):
            L = batch_masks.size(1)
            # Standard causal (autoregressive) mask: tokens can only attend to themselves and previous tokens.
            causal_mask = torch.tril(torch.ones((L, L), dtype=torch.long))
            # Multiply by the sample's padding mask (expanded to 2D) to zero out padded positions.
            sample_mask = batch_masks[i].unsqueeze(0).expand(L, L)
            combined_sample_mask = causal_mask * sample_mask
            causal_masks.append(combined_sample_mask)
        combined_attention_mask = torch.stack(causal_masks)  # shape: (batch_size, max_len, max_len)

        return batch_tokens, combined_attention_mask
