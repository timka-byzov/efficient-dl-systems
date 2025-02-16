import random
from torch.utils.data import Dataset, DataLoader, Sampler

from dataset.big_brain import BigBrainDataset


class UltraBigBrainDataset(BigBrainDataset):
    pass


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, k):

        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k

        # Build a hash table mapping each sample length to a list of indices.
        self.len_to_indices = {}
        for idx in range(len(dataset)):
            sample_length = len(dataset[idx])
            if sample_length not in self.len_to_indices:
                self.len_to_indices[sample_length] = []
            self.len_to_indices[sample_length].append(idx)

        # Randomize the order within each bucket.
        for length in self.len_to_indices:
            random.shuffle(self.len_to_indices[length])

    def __iter__(self):
        """
        Yields:
            batch (list): A list of indices forming one batch. In each batch, the difference 
                          between the longest and shortest sample lengths is <= k.
        """
        # Create pointers for each bucket
        pointers = {length: 0 for length in self.len_to_indices}
        # Set of lengths that still have available indices
        nonempty = {length for length in self.len_to_indices if pointers[length] < len(self.len_to_indices[length])}

        # Continue until all buckets are exhausted
        while nonempty:
            # Randomly choose a starting length from available buckets
            start_length = random.choice(list(nonempty))
            batch = []
            # Allowed range: from start_length to start_length + k (inclusive)
            for length in range(start_length, start_length + self.k + 1):
                if length in self.len_to_indices:
                    while (pointers[length] < len(self.len_to_indices[length]) and 
                           len(batch) < self.batch_size):
                        batch.append(self.len_to_indices[length][pointers[length]])
                        pointers[length] += 1
                    if pointers[length] >= len(self.len_to_indices[length]):
                        nonempty.discard(length)
                if len(batch) >= self.batch_size:
                    break
            yield batch

    def __len__(self):
        """
        Returns an estimate of the number of batches per epoch.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
