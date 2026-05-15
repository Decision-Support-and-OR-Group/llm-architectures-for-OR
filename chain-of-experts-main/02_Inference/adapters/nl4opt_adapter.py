from datasets import load_dataset


class NL4OptAdapter:

    def __init__(self, split: str = "test"):
        self.dataset_name = "CardinalOperations/NL4OPT"
        self.split = split
        self.data = None
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.dataset_name} (split='{self.split}')")
        try:
            self.data = load_dataset(self.dataset_name, split=self.split)
            print("Dataset loaded")
        except Exception as e:
            print("Failed to load data:", e)
            self.data = None

    def get_data(self):
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
        return self.data

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        """Allow indexing like adapter[idx]."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
        return self.data[idx]
