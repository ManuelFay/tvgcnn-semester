import numpy as np


class MinibatchSource:
    def __init__(self, data, labels, repeat=False):
        self.data = data
        self.labels = labels
        self.indices = np.random.permutation(self.dataset_length)
        self.idx = 0
        self.repeat = repeat
        self.end = False

    @property
    def dataset_length(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        if self.end:
            if not self.repeat:
                return None, self.end
            else:
                self.end = False

        end_idx = self.idx + batch_size

        if end_idx < self.dataset_length:
            batch = [self._data_chunk(self.idx, end_idx), self._labels_chunk(self.idx, end_idx)]
        else:
            self.end = True
            batch = [self._data_chunk(start=self.idx), self._labels_chunk(start=self.idx)]
            if self.repeat:
                end_idx = end_idx % self.dataset_length
                self.indices = np.random.permutation(self.dataset_length)
                batch[0] = np.concatenate((batch[0], self._data_chunk(end=end_idx)), axis=0)
                batch[1] = np.concatenate((batch[1], self._labels_chunk(end=end_idx)), axis=0)
        self.idx = end_idx

        return batch, self.end

    def _labels_chunk(self, start=None, end=None):
        if start is None:
            return self.labels[self.indices[:end]]
        elif end is None:
            return self.labels[self.indices[start:]]
        else:
            return self.labels[self.indices[start:end]]

    def _data_chunk(self, start=None, end=None):
        if start is None:
            return self.data[self.indices[:end]]
        elif end is None:
            return self.data[self.indices[start:]]
        else:
            return self.data[self.indices[start:end]]

    def restart(self):
        self.end = False
        self.idx = 0