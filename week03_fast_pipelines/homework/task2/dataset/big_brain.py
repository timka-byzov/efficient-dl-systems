import torch
from torch.utils.data.dataset import Dataset


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        # self.max_length = max_length
        self.tokenizer = tokenizer

        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # input_ids = input_ids[: self.max_length]
        return torch.tensor(input_ids, dtype=torch.long)


def manual_collate_fn(batch: list, max_length: int = None) -> tuple:
    
    lengths = [t.size(0) for t in batch]
    max_batch_len = max(lengths)
    pad_to = min(max_batch_len, max_length) if max_length is not None else  max_batch_len
    padded_tensors = []
    for t in batch:
        pad_length = pad_to - t.size(0)
        if pad_length > 0:
            # For BERT-base-uncased, pad token id is 0.
            padded = torch.cat([t, torch.zeros(pad_length, dtype=t.dtype)])
        else:
            padded = t[:pad_to]
        padded_tensors.append(padded)
    batch_tensor = torch.stack(padded_tensors)
    return batch_tensor
