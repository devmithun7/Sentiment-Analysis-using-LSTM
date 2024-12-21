import torch.multiprocessing as mp
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

from train_model import run_training
from data_processing import process_data
from utils import setup_logger

import sys

def main(world_size = 4, use_main = False):
    '''
    Main function to process data and run the distributed training.

    Parameters:
    world_size (int): Number of processes to spawn.
    use_main (bool): Whether to use the main dataset or a subset.
    '''
    # Data Processing parameters
    main_data_processed, subset_data_processed = process_data()
    if use_main:
        train_data = main_data_processed
        test_data = subset_data_processed
    else:
        train_data, test_data = train_test_split(subset_data_processed, test_size=0.2, random_state=42)
    
    # Model parameters
    batch_size = 32
    embedding_dim = 400
    hidden_dim = 512
    n_layers = 2
    epochs = 3
    world_size = world_size
    seq_len = 200
    
    # Set up logging
    setup_logger(world_size)
    
    try:
        logging.info("Starting distributed training with the following parameters:")
        print("Starting distributed training with the following parameters:")
        logging.info(f"Sequence length: {seq_len}")
        print(f"Sequence length: {seq_len}")
        logging.info(f"Batch size: {batch_size}")
        print(f"Batch size: {batch_size}")
        logging.info(f"Number of epochs: {epochs}")
        print(f"Number of epochs: {epochs}")
        logging.info(f"World size: {world_size}")
        print(f"World size: {world_size}")

        start_time = datetime.now()
        
        mp.spawn(
            run_training,
            args=(
                world_size,
                train_data,
                test_data,
                seq_len,
                batch_size,
                embedding_dim,
                hidden_dim,
                n_layers,
                epochs
            ),
            nprocs=world_size,
            join=True
        )
        
        delta = datetime.now() - start_time
        
        logging.info("Training completed successfully")
        print("Training completed successfully")
        logging.info(f"Time Taken for Training: {delta} or {delta.total_seconds()} seconds")
        print(f"Time Taken for Training: {delta} or {delta.total_seconds()} seconds")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        print(f"Training failed with error: {str(e)}")
        raise
    finally:
        print("Training process finished")

if __name__ == "__main__":
    print("Started Main Execution")
    if len(sys.argv) > 1:
        world_size = int(sys.argv[1])
    else:
        world_size = int(input("Enter Number of CPUs: "))
    main(world_size=world_size)