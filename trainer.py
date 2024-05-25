import time
import torch
import torch.nn.functional as F
from model import LanguageModel
from model_config import ModelConfig
from data_loader import ShaksphereDataLoader
from random import uniform
from IPython.display import display, clear_output
from torchsummaryX import summary

with open('input.txt', 'r') as f:
    text = f.read()
vocabulary = sorted(list(set(text)))
vocab_size = len(vocabulary)

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

model = LanguageModel()
train_loader = ShaksphereDataLoader(train_data, seq_len = ModelConfig['seq_len'], batch_size = ModelConfig['batch_size'])
val_loader = ShaksphereDataLoader(val_data, seq_len = ModelConfig['seq_len'], batch_size = ModelConfig['batch_size'])

class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = 0
        self.optimizer = optimizer
        self.criterion = criterion
        train_loss = 0


    def train(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            logits, attention_weights = self.model.forward(x, return_attention_weights=True)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = self.criterion(logits, y)
            total_loss = total_loss + loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        epoch_loss = total_loss / len(train_loader)
        self.epoch = self.epoch + 1
        print(f"Epoch: {self.epoch}\nTrain Loss: {epoch_loss}, Train Perplexity: {np.exp(epoch_loss)}, LR: {self.optimizer.param_groups[0]['lr']}")
        return epoch_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for x, y in self.val_loader:
            logits = self.model.forward(x)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            val_loss = self.criterion(logits, y)
            total_loss = total_loss + val_loss.item()

        epoch_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {epoch_loss}')
        return epoch_loss

    @torch.no_grad()
    def generate(self, max_tokens, prompt, temperature = 1, top_p = 1):
        self.model.eval()
        for i in range(max_tokens):
            logits = self.model.forward(prompt)
            logit = logits[:, -1, :] # (B, C)
            logit = logit / temperature
            probs = F.softmax(logit, dim = -1)
            weighted_probs = self.topPTransform(probs, top_p)
            token = torch.multinomial(weighted_probs, num_samples = 1) # (B, 1)
            prompt = torch.cat((prompt, token), dim = -1) # (B, T + 1)
        return prompt

    def topPTransform(self, probs, top_p):
        probs_sorted_vals, probs_sort_idx = torch.sort(probs, descending=True)
        prob_cumsum = torch.cumsum(probs_sorted_vals, dim = -1)

        absolute_diff = torch.abs(prob_cumsum - top_p)
        closest_index = torch.argmin(absolute_diff).item()
        idx_to_remove = probs_sort_idx[:, closest_index + 1:]

        mask = torch.ones_like(probs)
        mask[:, idx_to_remove] = 0

        probs = probs * mask
        weighted_probs = probs / torch.sum(probs, dim = -1)
        return weighted_probs

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved at " + path)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
optimizer.param_groups[0]['lr'] = 5e-5

epochs = 10
trainer = Trainer(model = model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion = criterion)
best_val_loss = float('inf')
print("Beginning the trainer")
for epoch in range(epochs):
    train_loss = trainer.train()
    val_loss = trainer.validate()
    #scheduler.step(val_loss)
    if val_loss < best_val_loss:
        trainer.save('./ShakGPT_3_512.pt')
        best_val_loss = val_loss
    print("Generation: " + decode(trainer.generate(prompt =
                            torch.tensor(encode("BARNARDO: Who's there?\nFRANCISCO: Nay, answer me. Stand and unfold yourself.\nBARNARDO:")).view(1, -1).to(device),
                            max_tokens=170,
                            top_p = 0.7)[0].tolist()))
    print('\n')
    if epoch == 700:
        optimizer.param_groups[0]['lr'] = 1e-5        