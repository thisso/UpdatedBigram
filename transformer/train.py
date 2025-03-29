import urllib.request
import torch
import  torch.nn as nn 
import torch.nn.functional as F

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "input.txt"

urllib.request.urlretrieve(url, filename)

#data exploration
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
# print(f"Length of dataset in characters: {len(data)}")
# print(data[:1000])

#sorted set of characters and possible elements 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

#2 encoding and decoding - tokenization
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# print(encode("Hii there"))
# print(decode(encode("Hii there")))

#3 creating data tensor w torch 
data = torch.tensor(encode(text), dtype = torch.long)
# print(data.shape)
# print(data[:1000])

#4 split up training data and validation data
n = int(0.9*len(data)) #first 90% will be trianing data  
train_data = data[:n]
val_data = data[n:]

#build context window size 
block_size = 8 #process 8 token at once 
train_data [:block_size+1] #+1 is important to give auto regressive properties 

#demonstration
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for i in range(block_size):
#     context = x[:i+1]
#     target = y[i]
#     print(f"when input is {context} the target: {target}")

#5 implementing batch dimensions 
torch.manual_seed(1337) #choosing a seed gets consistency for RNG in torch
#controls randomness not position
batch_size = 4 # how many sequences to process in parallel 

#stack 1D tensors into 4 by 8 matrix 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train') #getting a batch of training data and asisgning 2 variables 
# print('input: ')
# print(xb.shape)
# print(xb)
# print('target:')
# print(yb.shape)
# print(yb)

# print('----------')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target:{target}")

#6 implment a b-gram NN, single layer 
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #token directly reads off the logits for the next token from a lookup table 
        #1 layer of the NN 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens): #dont care about variable 
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) #forward pass
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), 
                        max_new_tokens=100)[0].tolist()))

#optimize model
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(100): #bump up step size to increase reults 
    #sampel, a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() #optimizer adjust the weights 
print(loss)