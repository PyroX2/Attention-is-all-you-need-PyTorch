
import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_sentence_length, d_model):
        super(PositionalEncoding, self).__init__()

        '''
        Implementation of positional encoding using for loop. The slow way.
        
        self.pe = torch.zeros((max_sentence_length, d_model)) # Initalize positional encoding with zeros
        for pos in range(max_sentence_length):
            for i in range(int(d_model/2)):
                encoding = torch.tensor(pos / (10_000)**(2*i/d_model))
                self.pe[pos, 2*i] = torch.sin(encoding)
                self.pe[pos, 2*i+1] = torch.cos(encoding)
        '''
        
        # Torch version
        positions = torch.arange(0, max_sentence_length).unsqueeze(1) # Create positions from 0 to max_sentence_length 
        positions = positions.expand(max_sentence_length, d_model) # Expand its dims so it can be divided by torch with embedding dim
        
        embedding_positions = torch.arange(0, d_model)
        
        div_term = (10_000)**(2*embedding_positions/d_model) # Create division terms for each embedding value
        
        self.pe = torch.zeros((max_sentence_length, d_model)) # Initialize positional encoding tensor with zeros
        
        # Apply positional encoding
        self.pe[:, 0::2] = torch.sin(positions[:, 0::2] / div_term[0::2])
        self.pe[:, 1::2] = torch.cos(positions[:, 1::2] / div_term[1::2])
        
                
    def forward(self, input):
        return input + self.pe

directory_length = 10_000
d_model = 512 # Model dimension, embedding dim
number_of_data_samples = 64
max_sentence_length = 128

# Create source and target data
source_data = torch.randint(1, directory_length, (number_of_data_samples, max_sentence_length))
target_data = torch.randint(1, directory_length, (number_of_data_samples, max_sentence_length))

embedding = torch.nn.Embedding(directory_length, d_model)

source_embedding = embedding(source_data)
target_embedding = embedding(target_data)

positional_encoding = PositionalEncoding(max_sentence_length, d_model)   

positional_encoding.forward(source_embedding).shape


