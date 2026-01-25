"""MS MARCO dataset loader and preprocessor."""


def load_and_prepare_data(dataset, tokenizer):
    """Load MS MARCO dataset and prepare training/validation examples."""
    print("Loading MS MARCO dataset...")
    
    # Prepare training examples
    train_examples = []
    for example in dataset['train']:
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        answer = example['answers'][0]
        
        # Skip unanswerable questions (MS MARCO has many "No Answer Present." examples)
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        # Concatenate all context passages (not just the first one)
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        train_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    print(f"✓ Loaded {len(train_examples)} training examples\n")
    
    # Prepare validation examples
    val_examples = []
    for example in dataset['validation']:
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        answer = example['answers'][0]
        
        # Skip unanswerable questions (MS MARCO has many "No Answer Present." examples)
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        # Concatenate all context passages (not just the first one)
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        val_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    print(f"✓ Loaded {len(val_examples)} validation examples\n")
    
    return train_examples, val_examples
