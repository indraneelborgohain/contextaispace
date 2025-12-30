"""
data_loader_context.py - Data loader that preserves document boundaries for context-aware training
"""

import torch
import gc
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from architecture.tokenizer import get_tokenizer

class DocumentDataset(Dataset):
    """
    Dataset that keeps document boundaries intact.
    Each sample is a full document (story) from TinyStories.
    """
    def __init__(self, documents, tokenizer, max_length=None):
        """
        Args:
            documents: List of text documents
            tokenizer: Tokenizer instance
            max_length: Deprecated - kept for compatibility but not used.
                        Model handles chunking via sliding_window in forward pass.
        """
        self.tokenizer = tokenizer
        self.samples = []
        
        print(f"Tokenizing {len(documents)} documents...")
        for doc in tqdm(documents):
            # Tokenize each document - keep entire document intact
            tokens = tokenizer.encode(doc)
            
            # Only skip documents that are too short
            if len(tokens) > 1:  # Need at least 2 tokens (input + target)
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Created {len(self.samples)} documents (model will handle chunking)")

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
    max_length=None,
    num_workers=4,
    shuffle_train=True
):
    """
    Create dataloaders that preserve document boundaries for context-aware training.
    
    Note: Documents are kept intact. The model's forward() method handles chunking
    based on its sliding_window parameter.
    
    Args:
        batch_size: Batch size
        max_length: Deprecated - kept for compatibility but not used
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
    
    # Create datasets - no max_length, model handles chunking
    train_dataset = DocumentDataset(train_docs, tokenizer)
    val_dataset = DocumentDataset(val_docs, tokenizer)
    
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
    print("Note: Documents are kept intact. Model will handle chunking via sliding_window.\n")
    train_loader, val_loader = create_context_dataloaders(
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )
    
    # Get tokenizer for decoding
    tokenizer = get_tokenizer()
    
    print("\nTesting train_loader with detailed debug output...")
    print("=" * 80)
    
    for batch_idx, (inputs, targets, doc_starts, lengths) in enumerate(train_loader):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx}")
        print(f"{'='*80}")
        print(f"Batch input shape: {inputs.shape}")
        print(f"Batch target shape: {targets.shape}")
        print(f"Document starts: {doc_starts.tolist()}")
        print(f"Sequence lengths: {lengths.tolist()}")
        
        # Examine each sequence in the batch
        for seq_idx in range(inputs.shape[0]):
            print(f"\n{'-'*80}")
            print(f"Sequence {seq_idx} in batch:")
            print(f"  Is document start: {doc_starts[seq_idx].item()}")
            print(f"  Length: {lengths[seq_idx].item()}")
            
            seq_input = inputs[seq_idx]
            seq_target = targets[seq_idx]
            seq_len = lengths[seq_idx].item()
            
            # Get actual tokens (without padding)
            actual_input = seq_input[:seq_len]
            actual_target = seq_target[:seq_len]
            
            print(f"\n  Input tokens (first 30): {actual_input[:30].tolist()}")
            print(f"  Target tokens (first 30): {actual_target[:30].tolist()}")
            
            # Verify input[i+1] == target[i]
            print(f"\n  Verification: input[1:] should match target[:-1]")
            if seq_len > 1:
                matches = (actual_input[1:] == actual_target[:-1]).all()
                print(f"  Match: {matches.item()}")
                if not matches.item():
                    print("  WARNING: Input/target mismatch detected!")
            
            # Decode text
            print(f"\n  Decoded input text (first 200 chars):")
            input_text = tokenizer.decode(actual_input.tolist())
            print(f"  {repr(input_text[:200])}")
            
            print(f"\n  Decoded target text (first 200 chars):")
            target_valid = actual_target[actual_target != -100]
            if len(target_valid) > 0:
                target_text = tokenizer.decode(target_valid.tolist())
                print(f"  {repr(target_text[:200])}")
            
            # Show the offset
            print(f"\n  Token offset demonstration (first 10 positions):")
            print(f"  Position | Input Token ID | Input Token Text         | Target Token ID | Target Token Text")
            print(f"  {'-'*100}")
            for pos in range(min(10, seq_len)):
                inp_tok = actual_input[pos].item()
                tgt_tok = actual_target[pos].item()
                
                # Decode individual tokens
                inp_text = tokenizer.decode([inp_tok])
                tgt_text = tokenizer.decode([tgt_tok]) if tgt_tok != -100 else "<PAD>"
                
                # Truncate text if too long
                inp_text = inp_text[:20] if len(inp_text) <= 20 else inp_text[:17] + "..."
                tgt_text = tgt_text[:20] if len(tgt_text) <= 20 else tgt_text[:17] + "..."
                
                print(f"  {pos:8d} | {inp_tok:14d} | {repr(inp_text):24s} | {tgt_tok:15d} | {repr(tgt_text):20s}")
        
        print(f"\n{'='*80}")
        
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    print("\nDataloader test complete!")
