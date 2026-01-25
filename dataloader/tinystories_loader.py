"""TinyStories dataset loader and preprocessor."""


def load_and_prepare_data(dataset, tokenizer, max_length=2048):
    """
    Load TinyStories dataset and prepare training/validation examples.
    
    For TinyStories, we use the entire story as context and train the decoder
    to generate the story autoregressively.
    
    Args:
        dataset: HuggingFace dataset object
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum story length in tokens
    
    Returns:
        train_examples: List of dicts with 'context' (story text)
        val_examples: List of dicts with 'context' (story text)
    """
    print("Loading TinyStories dataset...")
    
    # Prepare training examples
    train_examples = []
    for example in dataset['train']:
        if not example.get('text'):
            continue
        
        story = example['text'].strip()
        
        # Skip empty stories
        if len(story) == 0:
            continue
        
        # Skip very short stories (likely incomplete)
        if len(story) < 50:
            continue
        
        # Optionally truncate very long stories
        story_tokens = tokenizer.encode(story)
        if len(story_tokens) > max_length:
            # Truncate to max_length tokens
            story_tokens = story_tokens[:max_length]
            story = tokenizer.decode(story_tokens)
        
        train_examples.append({
            'context': story,
            'text': story  # Same as context for compatibility
        })
    
    print(f"✓ Loaded {len(train_examples)} training stories\n")
    
    # Prepare validation examples
    val_examples = []
    if 'validation' in dataset:
        for example in dataset['validation']:
            if not example.get('text'):
                continue
            
            story = example['text'].strip()
            
            # Skip empty stories
            if len(story) == 0:
                continue
            
            # Skip very short stories
            if len(story) < 50:
                continue
            
            # Optionally truncate very long stories
            story_tokens = tokenizer.encode(story)
            if len(story_tokens) > max_length:
                story_tokens = story_tokens[:max_length]
                story = tokenizer.decode(story_tokens)
            
            val_examples.append({
                'context': story,
                'text': story
            })
    
    print(f"✓ Loaded {len(val_examples)} validation stories\n")
    
    return train_examples, val_examples
