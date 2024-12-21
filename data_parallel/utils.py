import nltk
import numpy as np
import logging
from datetime import datetime
import os
import nltk
from collections import Counter
import numpy as np
import os
import logging
from datetime import datetime

def tokenize_and_build_vocab(sentences, min_occurrence_threshold=1):
    '''
    Tokeinze sentences and build vocabulary.
    Args:
        sentences: list of sentences
        min_occurrence_threshold: minimum occurrence threshold for words to be included in vocabulary
    Returns:
        tokenized_sentences: list of tokenized sentences
        word2idx: dictionary mapping words to indices
        idx2word: dictionary mapping indices to words
    '''
    words = Counter()
    tokenized_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokens)
        words.update([word.lower() for word in tokens])
    words = {k: v for k, v in words.items() if v > min_occurrence_threshold}
    words = sorted(words, key=words.get, reverse=True)
    words = ["_PAD", "_UNK"] + words
    word2idx = {o: i for i, o in enumerate(words)}
    idx2word = {i: o for i, o in enumerate(words)}
    return tokenized_sentences, word2idx, idx2word

def convert_sentences_to_indices(sentences, word2idx):
    '''
    Convert sentences to indices.
    Args:
        sentences: list of sentences
        word2idx: dictionary mapping words to indices
    Returns:
        list of sentences converted to indices
    '''
    return [[word2idx.get(word.lower(), 1) for word in sentence] for sentence in sentences]

def pad_input(sentences, seq_len):
    '''
    pads sentences to a fixed length.
    Args:
        sentences: list of sentences
        seq_len: fixed length to pad sentences to
    Returns:
        features: numpy array of padded sentences
    '''
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for idx, review in enumerate(sentences):
        if len(review) != 0:
            features[idx, -len(review):] = np.array(review)[:seq_len]
    return features

def split_validation_test(sentences, labels, split_frac):
    '''
    split sentences and labels into validation and test sets.
    Args:
        sentences: list of sentences
        labels: list of labels
        split_frac: fraction of data to use for validation
    Returns:
        val_sentences: list of validation sentences
        val_labels: list of validation labels
        test_sentences: list of test sentences
        test_labels: list of test labels
    '''
    split_idx = int(len(sentences) * split_frac)
    val_sentences, test_sentences = sentences[:split_idx], sentences[split_idx:]
    val_labels, test_labels = labels[:split_idx], labels[split_idx:]
    return val_sentences, val_labels, test_sentences, test_labels

def setup_logger(world_size):
    '''
    Set up logger.
    Args:
        world_size: number of CPUs used for training
    '''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{world_size}CPUs_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )