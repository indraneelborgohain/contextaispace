"""
data_loader_context.py - Data loader for encoder-decoder training with TinyStories and MS MARCO
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


class EncoderDecoderDataset(Dataset):
    """
    Dataset for encoder-decoder training.
    Each sample contains context, question, and answer.
    For TinyStories: context=first half, answer=second half, question=empty
    For MS MARCO: context=passages, question=query, answer=answer
    """
    def __init__(self, examples, tokenizer):
        """
        Args:
            examples: List of dicts with 'context', 'question', 'answer' keys
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
        self.examples = examples
        
        print(f"Created dataset with {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns:
            example: Dict with 'context', 'question', 'answer'
        """
        return self.examples[idx]


def collate_fn_encoder_decoder(batch):
    """
    Custom collate function for encoder-decoder training.
    Returns raw examples without batching - train_clean.py handles batching.
    
    Args:
        batch: List of example dicts
    
    Returns:
        batch: List of example dicts (unchanged)
    """
    return batch


def create_context_dataloaders(
    batch_size=1,
    num_workers=0,
    shuffle_train=True,
    max_tinystories=10000,
    max_msmarco=5000
):
    """
    Create dataloaders for encoder-decoder training with TinyStories and MS MARCO.
    
    Args:
        batch_size: Batch size (recommend 1 for encoder-decoder with gradient accumulation)
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
        max_tinystories: Maximum number of TinyStories examples to load (None for all)
        max_msmarco: Maximum number of MS MARCO examples to load (None for all)
    
    Returns:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
    """
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Load MS MARCO
    print("\n" + "="*60)
    print("Loading MS MARCO dataset...")
    print("="*60)
    msmarco_dataset = load_dataset("ms_marco", "v2.1")
    
    # Process MS MARCO
    msmarco_train = []
    for idx, example in enumerate(msmarco_dataset['train']):
        if max_msmarco and idx >= max_msmarco:
            break
        
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        answer = example['answers'][0]
        
        # Skip unanswerable questions
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        # Concatenate all context passages
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        msmarco_train.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    msmarco_val = []
    for idx, example in enumerate(msmarco_dataset['validation']):
        if max_msmarco and idx >= max_msmarco // 10:  # 10% of max for validation
            break
        
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        answer = example['answers'][0]
        
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        msmarco_val.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    print(f"✓ Loaded {len(msmarco_train)} MS MARCO training examples")
    print(f"✓ Loaded {len(msmarco_val)} MS MARCO validation examples")
    
    # Load TinyStories
    print("\n" + "="*60)
    print("Loading TinyStories dataset...")
    print("="*60)
    tinystories_dataset = load_dataset("roneneldan/TinyStories")
    
    # Process TinyStories - split each story in half
    tinystories_train = []
    for idx, example in enumerate(tinystories_dataset['train']):
        if max_tinystories and idx >= max_tinystories:
            break
        
        if not example.get('text'):
            continue
        
        story = example['text'].strip()
        
        if len(story) < 50:  # Skip very short stories
            continue
        
        # Split story roughly in half: first half = context, second half = answer
        story_tokens = tokenizer.encode(story, allowed_special='all')
        mid_point = len(story_tokens) // 2
        
        context_tokens = story_tokens[:mid_point]
        answer_tokens = story_tokens[mid_point:]
        
        # Decode back to text
        context = tokenizer.decode(context_tokens)
        answer = tokenizer.decode(answer_tokens)
        
        tinystories_train.append({
            'context': context,
            'question': '',  # No question for stories
            'answer': answer
        })
    
    tinystories_val = []
    for idx, example in enumerate(tinystories_dataset['validation']):
        if max_tinystories and idx >= max_tinystories // 10:  # 10% of max for validation
            break
        
        if not example.get('text'):
            continue
        
        story = example['text'].strip()
        
        if len(story) < 50:
            continue
        
        story_tokens = tokenizer.encode(story, allowed_special='all')
        mid_point = len(story_tokens) // 2
        
        context_tokens = story_tokens[:mid_point]
        answer_tokens = story_tokens[mid_point:]
        
        context = tokenizer.decode(context_tokens)
        answer = tokenizer.decode(answer_tokens)
        
        tinystories_val.append({
            'context': context,
            'question': '',
            'answer': answer
        })
    
    print(f"✓ Loaded {len(tinystories_train)} TinyStories training examples")
    print(f"✓ Loaded {len(tinystories_val)} TinyStories validation examples")
    
    # Combine datasets
    train_examples = msmarco_train + tinystories_train
    val_examples = msmarco_val + tinystories_val
    
    print("\n" + "="*60)
    print(f"TOTAL: {len(train_examples)} training examples, {len(val_examples)} validation examples")
    print(f"  - MS MARCO train: {len(msmarco_train)}, val: {len(msmarco_val)}")
    print(f"  - TinyStories train: {len(tinystories_train)}, val: {len(tinystories_val)}")
    print("="*60 + "\n")
    
    # Cleanup
    del msmarco_dataset, tinystories_dataset
    gc.collect()
    
    # Create datasets
    train_dataset = EncoderDecoderDataset(train_examples, tokenizer)
    val_dataset = EncoderDecoderDataset(val_examples, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn_encoder_decoder,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_encoder_decoder,
        pin_memory=True
    )
    
    return train_loader, val_loader


# If run as script, create and test the dataloaders
if __name__ == "__main__":
    print("Creating encoder-decoder dataloaders...")
    train_loader, val_loader = create_context_dataloaders(
        max_tinystories=100,
        max_msmarco=50,
        batch_size=2
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    print(f"\nSample training batch:")
    print("="*60)
    
    # Get first batch
    batch = next(iter(train_loader))
    
    for i, ex in enumerate(batch):
        print(f"\nExample {i+1}:")
        dataset_type = "MS MARCO" if ex['question'] else "TinyStories"
        print(f"Type: {dataset_type}")
        print(f"Context: {ex['context'][:150]}...")
        print(f"Question: {ex['question'] if ex['question'] else '(empty - story continuation)'}")
        print(f"Answer: {ex['answer'][:100]}...")
    
    print("\n" + "="*60)
    print("Dataloader test complete!")
