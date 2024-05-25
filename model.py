import time
import torch
import torch.nn.functional as F
from model_config import ModelConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.mini_head_size = int(self.head_size / self.num_heads)

        self.query = torch.nn.Linear(embed_dim, self.head_size)
        self.key = torch.nn.Linear(embed_dim, self.head_size)
        self.value = torch.nn.Linear(embed_dim, self.head_size)
        self.up_project = torch.nn.Linear(self.num_heads * self.mini_head_size, head_size)
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)

    def forward(self, tok_embeddings, return_attention_weights = False):
        B, T, C = tok_embeddings.shape
        Q = self.query(tok_embeddings)
        K = self.key(tok_embeddings)
        V = self.value(tok_embeddings)

        # Reshape into N sub heads for parallel processing
        mini_Q = Q.view(B, T, self.num_heads, self.mini_head_size).permute(0, 2, 1, 3) # (B, nh, T, mini_head)
        mini_K = K.view(B, T, self.num_heads, self.mini_head_size).permute(0, 2, 1, 3) # (B, nh, T, mini_head)
        mini_V = V.view(B, T, self.num_heads, self.mini_head_size).permute(0, 2, 1, 3) # (B, nh, T, mini_head)

        # Scaled dot Product
        affinities = (mini_Q @ mini_K.transpose(-1, -2)) * (self.mini_head_size ** -0.5) #  (B, nh, T, T)
        # Makes it causal decoder
        affinities = affinities.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attention_weights = F.softmax(affinities, dim = -1) # (B, nh, T, T)

        # weighted vlaues for each head
        mini_head_weighted_values = attention_weights @ mini_V # (B, nh, T, mini_head)
        # Concatenating each mini_head outputs
        mini_head_weighted_values = mini_head_weighted_values.permute(0, 2, 1, 3)
        head_weighted_values = mini_head_weighted_values.reshape(B, T, self.mini_head_size * self.num_heads) # (B, T, head_size)

        if return_attention_weights:
            return self.up_project(head_weighted_values), attention_weights
        return self.up_project(head_weighted_values) # (B, T, head_size)

class DecoderBlock(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=model_size)
        self.layer_norm_1 = torch.nn.LayerNorm(model_size)
        self.linear_1 = torch.nn.Linear(model_size, model_size)
        self.linear_2 = torch.nn.Linear(model_size, model_size)
        self.relu = torch.nn.ReLU()
        self.layer_norm_2 = torch.nn.LayerNorm(model_size)
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.dropout_2 = torch.nn.Dropout(0.1)

    def forward(self, embeddings, return_attention_weights = False):
        # tokens ---> (B, T, embed_size)
        B, T, C = embeddings.shape

        # Attention
        if return_attention_weights:
            weighted_values, attention_weights = self.attention.forward(embeddings, return_attention_weights)
        else:
            weighted_values = self.attention.forward(embeddings) # (B, T, embed_dim)

        # Attention Norm + dropout
        weighted_values_drp = self.dropout_1(weighted_values)
        norm_values = self.layer_norm_1(weighted_values_drp + embeddings) # (B, T, embed_dim)

        # FFN + dropout
        linear_1 = self.linear_1(norm_values)
        act_vals = self.relu(linear_1)
        linear_2 = self.linear_2(act_vals)
        linear_2_drp = self.dropout_2(linear_2)

        # LayerNorm
        ffn_norm = self.layer_norm_2(norm_values + linear_2_drp) # (B, T, embed_dim)


        if return_attention_weights:
            return ffn_norm, attention_weights
        return ffn_norm # (B, T, embed_dim)

class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = torch.nn.Embedding(seq_len, embed_dim)
        self.dropout_0 = torch.nn.Dropout(0.1)
        self.decoder_block_1 = DecoderBlock(num_heads=8)
        self.decoder_block_2 = DecoderBlock(num_heads=8)
        self.decoder_block_3 = DecoderBlock(num_heads=8)

        self.projection = torch.nn.Linear(model_size, vocab_size)
        self.projection.weight = self.embedding_layer.weight

    def forward(self, tokens, return_attention_weights = False):
        B, T = tokens.shape
        # Token and Positional Embeddings + dropout
        tok_embs = self.embedding_layer(tokens) # (B, T, embed_dim)
        pos_embs = self.positional_embedding(torch.arange(T).to(device))
        pos_tok_embs = tok_embs + pos_embs # (B, T, embed_dim)
        pos_tok_embs_drp = self.dropout_0(pos_tok_embs)

        # Decoder Blocks
        decoder_1 = self.decoder_block_1.forward(pos_tok_embs)
        decoder_2 = self.decoder_block_2.forward(decoder_1)
        if return_attention_weights:
            decoder_3, last_layer_attention_weights = self.decoder_block_3.forward(decoder_2, return_attention_weights=True)
        else:
            decoder_3 = self.decoder_block_3.forward(decoder_2)

        # projection layer
        logits = self.projection(decoder_3) # (B, T, vocab_size)

        if return_attention_weights:
            return logits, last_layer_attention_weights
        return logits # (B, T, vocab_size)