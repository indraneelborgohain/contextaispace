
import torch,gc
from torch.utils.data import Dataset,DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
batch_size=5
context_len=4000

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = get_tokenizer()
print("tokenizing...")

class StoryDataset(Dataset):
    def __init__(self, stories):
        self.input_ids = []
        self.target_ids = []
        for story in tqdm(stories):
            tokens = tokenizer.encode(story["text"])
            if len(tokens) > 1:  # Need at least 2 tokens for input/target
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                self.input_ids.append(torch.tensor(input_tokens, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_tokens, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

train_dataset = StoryDataset(dataset['train'])
val_dataset = StoryDataset(dataset['validation'])
print("tokenized")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

del dataset
gc.collect()