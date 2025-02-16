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

                tokens = self.tokenizer.tokenize(line)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                token_tensor = torch.tensor(token_ids, dtype=torch.long)

                # If the sample is too long, either drop it or truncate it.
                if token_tensor.size(0) > self.max_length:
                    if self.drop_long_samples:
                        continue
                    else:
                        token_tensor = token_tensor[:self.max_length]

                sample_attention = torch.ones(token_tensor.size(0), dtype=torch.long)

                batch_token_tensors.append(token_tensor)
                batch_attention_masks.append(sample_attention)

                if len(batch_token_tensors) == self.batch_size:
                    batch_tokens, combined_attention_mask = self.pad_and_build_mask(
                        batch_token_tensors, batch_attention_masks
                    )
                    yield batch_tokens, combined_attention_mask
                    batch_token_tensors = []
                    batch_attention_masks = []

            if batch_token_tensors:
                batch_tokens, combined_attention_mask = self.pad_and_build_mask(
                    batch_token_tensors, batch_attention_masks
                )
                yield batch_tokens, combined_attention_mask

    def pad_and_build_mask(self, tokens_list, masks_list):

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

        causal_masks = []
        for i in range(batch_tokens.size(0)):
            L = batch_masks.size(1)
            causal_mask = torch.tril(torch.ones((L, L), dtype=torch.long))
            sample_mask = batch_masks[i].unsqueeze(0).expand(L, L)
            combined_sample_mask = causal_mask * sample_mask
            causal_masks.append(combined_sample_mask)
        combined_attention_mask = torch.stack(causal_masks)  # shape: (batch_size, max_len, max_len)

        return batch_tokens, combined_attention_mask
