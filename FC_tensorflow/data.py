class Dataset:
    def __init__(self, data):
        self.data = data

    def range(self, start, limit=None, delta=1):
        if limit is None:
            start, limit = 0, start
        return Dataset(list(range(start, limit, delta)))

    def as_numpy_iterator(self):
        return iter(self.data)



