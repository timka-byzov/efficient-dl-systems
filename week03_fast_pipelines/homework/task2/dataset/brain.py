import torch
from torch.utils.data.dataset import Dataset




class BrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 640):
        self.max_length = max_length

        self.tokenizer = tokenizer
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[: self.max_length]
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_length
        return torch.tensor(input_ids, dtype=torch.long)
