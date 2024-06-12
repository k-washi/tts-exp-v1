import torch
from .dataset import AudioTextDataset

def collate_fn(batch):
    """VITSモデルで必要なデータの形に変更
    Args:
        batch : (wav, spec, phonome_indexes, accent_indexes, spk_id)
    Returns:
        (
            wav_padded,
            wav_lengths,
            spec_padded,
            spec_lengths,
            text_padded,
            text_lengths,
            accent_pos_padded,
            spk_id,
        )
    """
    max_wav_len = max([x[0].size(1) for x in batch])
    max_spec_len = max([x[1].size(1) for x in batch])
    max_phonome_len = max([x[2].size(0) for x in batch])
    max_accent_len = max([x[3].size(0) for x in batch])
    assert max_phonome_len == max_accent_len
    
    batch_size = len(batch)
    wav_lengths = torch.LongTensor(batch_size)
    spec_lengths = torch.LongTensor(batch_size)
    phonome_lengths = torch.LongTensor(batch_size)
    
    wav_padded = torch.zeros(batch_size, 1, max_wav_len, dtype=torch.float32)
    spec_padded = torch.zeros(
        batch_size, batch[0][1].size(0), max_spec_len, dtype=torch.float32
    )
    phonome_padded = torch.zeros(batch_size, max_phonome_len, dtype=torch.long)
    accent_padded = torch.zeros(batch_size, max_accent_len, dtype=torch.long)
    spk_id_padded = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (
        wav_row,
        spec_row,
        p_row,
        a_row,
        spk_id,
    ) in enumerate(batch):
        wav_padded[i, :, : wav_row.size(1)] = wav_row
        spec_padded[i, :, : spec_row.size(1)] = spec_row
        phonome_padded[i, : p_row.size(0)] = p_row
        accent_padded[i, : a_row.size(0)] = a_row
        spk_id_padded[i] = spk_id

        wav_lengths[i] = wav_row.size(1)
        spec_lengths[i] = spec_row.size(1)
        phonome_lengths[i] = p_row.size(0)
    return (
        wav_padded,
        wav_lengths,
        spec_padded,
        spec_lengths,
        phonome_padded,
        phonome_lengths,
        accent_padded,
        spk_id_padded,
    )
        


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: AudioTextDataset,
        batch_size:int,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(
                    torch.randperm(len(bucket), generator=g).tolist()
                )
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size