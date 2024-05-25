import time
import torch
import torch.nn.functional as F
from model_config import ModelConfig
from model import LanguageModel
from random import uniform
from IPython.display import display, clear_output
from torchsummaryX import summary

seq_len = ModelConfig['seq_len']
batch_size = ModelConfig['batch_size']
embed_dim = ModelConfig['embed_dim']
qk_dim = ModelConfig['qk_dim']
value_dim = ModelConfig['embed_dim']
model_size = ModelConfig['embed_dim']

with open('input.txt', 'r') as f:
    text = f.read()
vocabulary = sorted(list(set(text)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_to_int = {v:i for i, v in enumerate(vocabulary)}
int_to_vocab = {i:v for i, v in enumerate(vocabulary)}

encode = lambda x: [vocab_to_int[i] for i in x]
decode = lambda x: "".join([int_to_vocab[i] for i in x])

model = LanguageModel()
model.to(device)
model.load_state_dict(torch.load('./ShakGPT_3_512.pt', map_location = device))

def topPTransform(probs, top_p):
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

def persistent_generation(prompt, max_tokens, temperature = 1, top_p = 1):
    generated_text = prompt
    model.eval()
    tokens = torch.tensor(encode(prompt)).reshape(1, -1).to(device)
    buffer = 10
    token_count = 0
    while token_count < max_tokens:
        if tokens.shape[-1] >= seq_len:
            tokens = tokens[:, tokens.shape[-1] - seq_len + 10: ]
        logits = model.forward(tokens)
        logit = logits[:, -1, :] # (B, C)

        # temperature
        logit = logit / temperature

        # top_p
        probs = F.softmax(logit, dim = -1) # (B, C)
        weighted_probs = topPTransform(probs, top_p)
        predicted_token = torch.multinomial(weighted_probs, num_samples = 1) # (B, 1)
        generated_text = generated_text + decode(predicted_token[0].cpu().detach().tolist())
        tokens = torch.cat((tokens, predicted_token), dim = -1) # (B, T + 1) # (1, 1)
        clear_output(wait=True)

        print(generated_text)
        token_count = token_count + 1

        #time.sleep(0.01)

def main():
    sentences = input("Enter the sentences: ")
    max_tokens = int(input("Enter the maximum number of tokens: "))
    top_p = float(input("Enter the top_p value: "))

    print(persistent_generation(sentences, max_tokens = max_tokens, top_p = top_p))

if __name__ == "__main__":
    main()