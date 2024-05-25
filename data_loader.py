import torch
from model_config import ModelConfig

with open('input.txt', 'r') as f:
    text = f.read()
vocabulary = sorted(list(set(text)))
vocab_size = len(vocabulary)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seq_len = ModelConfig['seq_len']
batch_size = ModelConfig['batch_size']
embed_dim = ModelConfig['embed_dim']
qk_dim = ModelConfig['qk_dim']
value_dim = ModelConfig['embed_dim']
model_size = ModelConfig['embed_dim']

vocab_to_int = {v:i for i, v in enumerate(vocabulary)}
int_to_vocab = {i:v for i, v in enumerate(vocabulary)}

encode = lambda x: [vocab_to_int[i] for i in x]
decode = lambda x: "".join([int_to_vocab[i] for i in x])

dataset = torch.tensor(encode(text), dtype = torch.long)
train_data = dataset[:int(0.9 * len(dataset))]
val_data = dataset[int(0.9 * len(dataset)): int(0.97 * len(dataset))]
test_data = dataset[int(0.97 * len(dataset)): ]

class ShaksphereDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, seq_len, batch_size):
        self.dataset = dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sub_seq = (len(dataset) // self.seq_len)
        self.num_batches = self.sub_seq // self.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        offsets = torch.randperm(len(self.dataset) - self.seq_len)[:self.sub_seq]
        inputs = torch.stack([self.dataset[i: i + self.seq_len] for i in offsets])
        outputs = torch.stack([self.dataset[i + 1: i + 1 + self.seq_len] for i in offsets])
        rem = inputs.shape[0] % self.batch_size
        if rem != 0:
            inputs = inputs[:-rem, :].reshape(self.num_batches, self.batch_size, self.seq_len)
            outputs = outputs[:-rem, :].reshape(self.num_batches, self.batch_size, self.seq_len)
        else:
            inputs = inputs.reshape(self.num_batches, self.batch_size, self.seq_len)
            outputs = outputs.reshape(self.num_batches, self.batch_size, self.seq_len)

        batch_idx = 0
        while batch_idx < self.num_batches:
            yield inputs[batch_idx, : , :].to(device), outputs[batch_idx, :, :].to(device)
            batch_idx = batch_idx + 1