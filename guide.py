import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
# src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

from collections import Counter

class Vocab:
    def __init__(self, tokens, min_freq=1, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        counter = Counter(token for sent in tokens for token in sent)
        self.itos = specials + [tok for tok, freq in counter.items() if freq >= min_freq and tok not in specials]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)

# Example usage
def tokenize_en(text): return text.lower().split()
def tokenize_zh(text): return list(text.strip())

# Load and parse the dataset
def load_parallel_data(path, max_samples=10000):
    src_sentences = []
    tgt_sentences = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '\t' not in line: continue
            eng, zh, *_ = line.strip().split('\t')
            src_sentences.append(eng.lower())
            tgt_sentences.append(zh)
            if len(src_sentences) >= max_samples:
                break
    return src_sentences, tgt_sentences

eng_texts, zh_texts = load_parallel_data("datasets/cmn.txt", max_samples=5000)

# def tokenize_en(sentence):
#     return sentence.lower().split()

# def tokenize_zh(sentence):
#     return list(sentence.strip())  # Character-level

en_tokens = [tokenize_en(s) for s in eng_texts]
zh_tokens = [tokenize_zh(s) for s in zh_texts]


en_vocab = Vocab(en_tokens)
zh_vocab = Vocab(zh_tokens)

print(f"English vocabulary size: {len(en_vocab)}")
print(f"Chinese vocabulary size: {len(zh_vocab)}")
print(f"English vocabulary: {en_vocab.itos[:10]}")
print(f"Chinese vocabulary: {zh_vocab.itos[:10]}")

def pad_batch(batch, pad_id):
    max_len = max(len(x) for x in batch)
    return torch.tensor([x + [pad_id] * (max_len - len(x)) for x in batch])

# src_vocab = build_vocab(eng_texts, tokenize_en)
# tgt_vocab = build_vocab(zh_texts, tokenize_zh)

src_batch = [en_vocab.encode(["<sos>"] + tokenize_en(s) + ["<eos>"]) for s in eng_texts]
tgt_batch = [zh_vocab.encode(["<sos>"] + tokenize_zh(s) + ["<eos>"]) for s in zh_texts]

src_batch_padded = pad_batch(src_batch, en_vocab.stoi["<pad>"])
tgt_batch_padded = pad_batch(tgt_batch, zh_vocab.stoi["<pad>"])

print("Source batch (padded):")
print(src_batch_padded)
print("Target batch (padded):")
print(tgt_batch_padded)

src_data = src_batch_padded
tgt_data = tgt_batch_padded

# Split data into train and validation sets
train_ratio = 0.9
train_size = int(len(src_data) * train_ratio)

src_train = src_data[:train_size]
tgt_train = tgt_data[:train_size]
src_val = src_data[train_size:]
tgt_val = tgt_data[train_size:]

from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
train_dataset = TensorDataset(src_train, tgt_train)
val_dataset = TensorDataset(src_val, tgt_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.stoi["<pad>"])
# optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
transformer.train()

for epoch in range(100):
    total_loss = 0
    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()
        output = transformer(src_batch, tgt_batch[:, :-1])
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")


transformer.eval()
val_loss = 0
with torch.no_grad():
    for src_batch, tgt_batch in val_loader:
        output = transformer(src_batch, tgt_batch[:, :-1])
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_batch[:, 1:].reshape(-1))
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
print(f"Validation Loss: {avg_val_loss:.4f}")

def translate_sentence(sentence, transformer, en_vocab, zh_vocab, max_len=50, device='cpu'):
    transformer.eval()
    
    # Tokenize and encode the English sentence
    tokens = ["<sos>"] + tokenize_en(sentence) + ["<eos>"]
    src_ids = en_vocab.encode(tokens)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)  # shape: (1, src_len)

    # Start decoding with <sos>
    tgt_ids = [zh_vocab.stoi["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)  # shape: (1, tgt_len)
        with torch.no_grad():
            output = transformer(src_tensor, tgt_tensor)  # (1, tgt_len, vocab_size)
        
        next_token = output[0, -1].argmax(dim=-1).item()
        tgt_ids.append(next_token)
        
        if next_token == zh_vocab.stoi["<eos>"]:
            break

    # Decode IDs to Chinese characters
    translated_tokens = zh_vocab.decode(tgt_ids[1:-1])  # remove <sos> and <eos>
    return ''.join(translated_tokens)

example_sentence = "help me!"
translation = translate_sentence(example_sentence, transformer, en_vocab, zh_vocab)
print(f"{example_sentence} → {translation}")