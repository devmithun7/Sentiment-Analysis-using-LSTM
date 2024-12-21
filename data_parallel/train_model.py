import nltk
import torch
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from datetime import datetime

from model import SentimentNet
from utils import tokenize_and_build_vocab, convert_sentences_to_indices, pad_input, split_validation_test
from evaluate_model import evaluate_model
import json

def setup(rank, world_size):
    '''
    Initialize the process group for distributed training.
    '''
    # Pick a free port
    port = str(12355 + rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    # Initialize process group without setting cuda device
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"[INFO] Rank {rank}/{world_size} process initialized on port {port}")

def cleanup():
    '''
    cleanup the process group after training.
    '''
    dist.destroy_process_group()

def save_model(model, save_dir, world_size, epochs, batch_size, embedding_dim, hidden_dim, n_layers):
    """
    Save both the model state dictionary and the entire model to the specified directory.
    """
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save state dictionary
    state_dict_path = os.path.join(save_dir, 'state_dict.pt')
    torch.save(model.module.state_dict(), state_dict_path)
    print(f"State dictionary saved at: {state_dict_path}")
    
    # Save entire model
    full_model_path = os.path.join(save_dir, 'full_model.pt')
    torch.save(model.module, full_model_path)
    print(f"Full model saved at: {full_model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'world_size': world_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers
    }
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved at: {metadata_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, batch_size, device, rank, world_size, embedding_dim, hidden_dim, n_layers, save_dir):
    '''
    Train the model using the provided data loaders, criterion, and optimizer.
    '''
    valid_loss_min = float('inf')
    
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        # Training phase
        counter = 0
        model.train()
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            h = model.module.init_hidden(inputs.size(0), device)
            h = tuple([e.data for e in h])
            
            optimizer.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            if counter % 10 == 0:
                print(f"epoch = {epoch}, rank = {rank}, loss = {loss.item() * inputs.size(0)}, count = {counter}")
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                h = model.module.init_hidden(inputs.size(0), device)
                h = tuple([e.data for e in h])
                
                output, h = model(inputs, h)
                loss = criterion(output.squeeze(), labels.float())
                valid_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(val_loader.dataset)
        
        if rank == 0:
            print(f'Epoch: {epoch+1}/{epochs}')
            print(f'Training Loss: {train_loss:.6f} Validation Loss: {valid_loss:.6f}')
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
                
                # Save model state dictionary and full model
                save_model(
                    model,
                    save_dir,
                    world_size=world_size,
                    epochs=epochs,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers
                )
                
                valid_loss_min = valid_loss
    
    return model

def run_training(rank, world_size, train_data, test_data, seq_len,
                batch_size, embedding_dim, hidden_dim, n_layers, epochs):
    '''
    train model for each rank
    '''
    setup(rank, world_size)

    save_dir = f'saved_models/version_{world_size}_CPUs'
    
    # Modify device selection
    device = torch.device('cpu')
    
    # Rest of the run_training function remains the same
    
    # Tokenize and prepare data
    train_sentences, word2idx, _ = tokenize_and_build_vocab(train_data['review'])
    test_sentences = [nltk.word_tokenize(sentence) for sentence in test_data['review']]
    
    train_sentences = convert_sentences_to_indices(train_sentences, word2idx)
    test_sentences = convert_sentences_to_indices(test_sentences, word2idx)
    
    train_sentences = pad_input(train_sentences, seq_len)
    test_sentences = pad_input(test_sentences, seq_len)
    
    train_labels = train_data['label'].astype(int).values
    test_labels = test_data['label'].astype(int).values
    
    val_sentences, val_labels, test_sentences, test_labels = split_validation_test(
        test_sentences, test_labels, 0.5
    )
    
    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
    val_dataset = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
    test_dataset = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))
    
    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    vocab_size = len(word2idx) + 1
    model = SentimentNet(vocab_size, 1, embedding_dim, hidden_dim, n_layers)
    model = model.to(device)
    
    # Modify DDP initialization
    model = DDP(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, batch_size, device, rank, world_size, embedding_dim, hidden_dim, n_layers, save_dir)

    # Test model if rank 0
    if rank == 0:
        evaluation_results = evaluate_model(model, test_loader, criterion, device)
        evaluation_path = os.path.join(save_dir, 'evaluation.json')
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation_results, f)
        print(f"evaluation results saved at: {evaluation_path}")
        
    
    cleanup()
    print("Done Training ------------------------------------")
    