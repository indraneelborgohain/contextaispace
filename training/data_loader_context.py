"""
data_loader_context.py - Data loader that preserves document boundaries for context-aware training
"""

import torch
import gc
from torch.utils.data import Dataset, DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
from tqdm import tqdm

class DocumentDataset(Dataset):
    """
    Dataset that keeps document boundaries intact.
    Each sample is a full document (story) from TinyStories.
    """
    def __init__(self, documents, tokenizer, max_length=4096):
        """
        Args:
            documents: List of text documents
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length per document
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Tokenizing {len(documents)} documents...")
        for doc in tqdm(documents):
            # Tokenize each document
            tokens = tokenizer.encode(doc)
            
            # If document is longer than max_length, split into chunks
            # Each chunk is treated as a separate "document" for context purposes
            if len(tokens) > max_length:
                for i in range(0, len(tokens) - 1, max_length):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) > 1:  # Need at least 2 tokens (input + target)
                        self.samples.append(torch.tensor(chunk, dtype=torch.long))
            else:
                if len(tokens) > 1:  # Need at least 2 tokens
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Created {len(self.samples)} document chunks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Token sequence (all but last token)
            target_ids: Target sequence (all but first token)
            is_doc_start: Boolean indicating if this is the start of a new document
        """
        tokens = self.samples[idx]
        # For context models, we always treat each sample as document start
        # This ensures context is reset between documents
        return tokens[:-1], tokens[1:], True


def collate_fn_with_padding(batch):
    """
    Custom collate function that pads sequences to the same length within a batch.
    
    Args:
        batch: List of (input_ids, target_ids, is_doc_start) tuples
    
    Returns:
        input_ids: Padded tensor (batch_size, max_seq_len)
        target_ids: Padded tensor (batch_size, max_seq_len)
        is_doc_start: Boolean tensor (batch_size,)
        lengths: Original lengths before padding (batch_size,)
    """
    inputs, targets, doc_starts = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([len(inp) for inp in inputs])
    max_len = lengths.max().item()
    
    # Pad sequences (use -100 as padding for targets to ignore in loss)
    padded_inputs = torch.stack([
        torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
        for inp in inputs
    ])
    
    padded_targets = torch.stack([
        torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=-100)
        for tgt in targets
    ])
    
    doc_starts_tensor = torch.tensor(doc_starts, dtype=torch.bool)
    
    return padded_inputs, padded_targets, doc_starts_tensor, lengths


def create_context_dataloaders(
    batch_size=5,
    max_length=4096,
    num_workers=4,
    shuffle_train=True
):
    """
    Create dataloaders that preserve document boundaries for context-aware training.
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length per document
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    # Load TinyStories dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Extract documents (each story is a separate document)
    train_docs = [ex["text"] for ex in dataset['train']]
    val_docs = [ex["text"] for ex in dataset['validation']]
    
    print(f"Loaded {len(train_docs)} training documents, {len(val_docs)} validation documents")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Create datasets
    train_dataset = DocumentDataset(train_docs, tokenizer, max_length=max_length)
    val_dataset = DocumentDataset(val_docs, tokenizer, max_length=max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_padding
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_padding
    )
    
    # Cleanup
    del dataset, train_docs, val_docs
    gc.collect()
    
    return train_loader, val_loader


# If run as script, create and test the dataloaders
if __name__ == "__main__":
    print("Creating context-aware dataloaders...")
    train_loader, val_loader = create_context_dataloaders(
        batch_size=2,
        max_length=512,
        num_workers=0  # Use 0 for testing
    )
    
    print("\nTesting train_loader...")
    for i, (inputs, targets, doc_starts, lengths) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Doc starts: {doc_starts}")
        print(f"  Lengths: {lengths}")
        if i >= 2:
            break
    
    print("\nDataloader test complete!")
