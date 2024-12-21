import torch.nn as nn
import torch

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, devices=None):
        '''
        constructor for Parallel Model
        '''
        super(SentimentNet, self).__init__()
        if devices is None:
            devices = ['cpu', 'cpu']  # Default devices are CPUs
        
        self.device0 = torch.device(devices[0])  # Device for embedding and output layers
        self.device1 = torch.device(devices[1])  # Device for LSTM layer

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Embedding layer on device 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device0)

        # LSTM layer on device 1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True).to(self.device1)

        # Dropout and fully connected layers on device 0
        self.dropout = nn.Dropout(drop_prob).to(self.device0)
        self.fc = nn.Linear(hidden_dim, output_size).to(self.device0)
        self.sigmoid = nn.Sigmoid().to(self.device0)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        # Step 1: Embedding on device 0
        x = x.to(self.device0)
        embeds = self.embedding(x)

        # Step 2: LSTM on device 1
        embeds = embeds.to(self.device1)  # Move embeddings to device 1
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Step 3: Fully connected layer on device 0
        lstm_out = lstm_out.to(self.device0)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        # Reshape output
        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state for LSTM. 
        The hidden state tensors must be placed on the same device as the LSTM layer (device1).
        """
        weight = next(self.lstm.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device1),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device1)
        )
        return hidden
