import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
#------------------------------

torch.manual_seed(1337)

with open("input.txt","r",encoding="utf-8") as f:
  text =  f.read()

#here are all the unique chracters that occur in the text file
chars = sorted(list (set(text))) # set gives the set of unique char then into list -> sorting
vocab_size = len(chars)

#for tokeninzation we are building encoder and decoder.
stoi  = {ch:i for i ,ch in enumerate(chars) } #str to index
itos = {i:ch for i,ch in enumerate(chars)} #index to str
encode =  lambda s: [stoi[c] for c in s] #encoder: fn take a str ,output a list of integers
decode =  lambda l: "".join([itos[i] for i in l]) #decoder: fn take a list of integers , output a str.

data = torch.tensor(encode(text),dtype = torch.long)
n =int(0.9*len(data)) #90% data
train_data = data[:n]
val_data  = data[n:] #this will help to understand at what extend tour model is overfitting.


def get_batch(split):
  #generate a small batch of data of inputs x and targets y
  data = train_data  if split == "train" else val_data
  ix = torch.randint(len(data)- block_size, (batch_size,))
  #generating random integer within the len range of strings or index.
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device),y.to(device)
# output of above line.
# tensor([[24, 43, 58,  5, 57,  1, 46, 43],
#         [44, 53, 56,  1, 58, 46, 39, 58],
#         [52, 58,  1, 58, 46, 39, 58,  1],
#         [25, 17, 27, 10,  0, 21,  1, 54]])

  return x,y

@torch.no_grad()
def estimate_loss():
  # this is the cost function to evaluate average loss on train and val sets Of whole model
  # venv
  out= {}
  model.eval()
  for split in ["train","val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x,y = get_batch(split)
      logits,loss = model(x,y)
      losses[k] = loss.item()
    out[f"{split}_loss"] = losses.mean().item()
  model.train()
  return out

class Head(nn.Module):
  """one head of self attention"""

  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embed,head_size,bias=False) #key is the input to the attention function
    self.query = nn.Linear(n_embed,head_size,bias=False) #query is the input to the attention function
    self.value = nn.Linear(n_embed,head_size,bias=False)
    self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size))) #this is cearting the lower tirangluar matrix for class.

    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
      B,T,C = x.shape #batch, time , channels

      # lets see a single head perform self attention
      k= self.key(x) #(B,T,16 )
      q= self.query(x) #(B,T,16)
      wei = q @ k.transpose(-2,-1)*(C**-0.5) # (B,T,16) @ (B,16,T) ---> (B,T,T) #dvie by root to mormalise the value inside matrix
      #extra c**-0.5 is to scale the dot product to avoid large values for more look into white peper
      wei = wei.masked_fill(self.tril[:T,:T] ==0, float("-inf"))
      wei = F.softmax(wei,dim= -1)
      # out = wei @ x

      v = self.value(x)
      out = wei @ v

      # torch.allclose(xbow,xbow3,atol=1e-6, rtol=1e-5)
      # problem is that if i am vowel then i look for consonants in past to to
      # get the info flow to me but in a data dependent way

      # this is what self attention solves.
      # how ?
      # every single token at each position will emit two vectors
      #  it will emit a query and it will emit a key
      # the query vector is speaking what am i looking for and the key vector explain what do i contain
      # then will do a dot product between key and query to set affinities between token.
      return out
  
class MultiHeadAttention(nn.Module):
  """multiple heads of self attention in parallel"""
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads= nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this is creating the list of heads
    self.proj = nn.Linear(n_embed,n_embed)
    self.dropout = nn.Dropout(dropout)


  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads],dim=-1)
    out = self.dropout(self.proj(out))
    return  out #this is concatenating the output of all the heads

class FeedForward(nn.Module):

  """a simple linear layer followed by a non-linearity"""
  def __init__(self,n_embed):
    super().__init__()
 
    self.net = nn.Sequential(  #here sequential take linear embedding as input and give it to relu to introduce non linearity and stacking the layers.
        nn.Linear(n_embed,4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed, n_embed),
        nn.Dropout(dropout),
    )

  def forward(self,x):
      return self.net(x)

class Block(nn.Module):
  """Transformer block: communication followed by computation"""

  def __init__(self , n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head

    self.sa = MultiHeadAttention(n_head,head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
  
  def forward(self,x):

    x =x + self.sa(self.ln1(x))  #here ln is layer normalisation done twice.
    x =x +  self.ffwd(self.ln2(x))
    return x



class BigramLanguageModel(nn.Module):  #inheriting parent class

  def __init__(self):
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table.
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding= nn.Embedding(block_size,n_embed)
    # self.sa_head = MultiHeadAttention(4,n_embed//4) #4 is the number of heads in the multi head attention
    # self.ffwd  = FeedForward(n_embed) #this is the feed forward layer which will take the output of the self attention and give the output of same shape
    self.blocks = nn.Sequential(
            *[Block(n_embed,n_head = n_head) for _ in range(n_layer)]
    )

    self.ln_f = nn.LayerNorm(n_embed)  #final layer
    self.lm_head =  nn.Linear(n_embed,vocab_size) #linear layer to get the logits
    

  def forward(self, idx ,targets=None):
    B,T = idx.shape #B is batch size and T is the sequence length
    #idx and targets are both (B,T) tensor of integers
    #BTC is batch time channel
    token_emb = self.token_embedding_table(idx) #(BTC) they are mapping of token in embedding matrix refer to step 15 in notes   
    pos_emb = self.position_embedding(torch.arange(T,device=device)) #(T,n_embed) this is the position embedding for each token in the sequence
    x = token_emb + pos_emb # (B,T,n_embed)  x not only hold the toke identity but also the position of the token in the sequence
    # x = self.sa_head(x) # (B,T,n_embed) this is the self attention head which will take the input and give the output of same shape
    x = self.blocks(x) # (B,T,C)

    logits = self.lm_head(x)  #B,T,C  above c and this c is not equal -> above c is embed_size and this c is vocab size
    if targets is None:
      loss = None
    else:
     
      #this reshaping is done due to cross_entropy functions input it take inputs as  (N,C) shape and target= N
      # n=  number of samples or batch size * sequence length(BxT) and c is  number of classes(vocab size)
      B,T,C = logits.shape
      logits = logits.view(B*T,C)  #changing the shape of logits
      targets =  targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    return logits,loss
  
  def generate(self,idx,max_new_tokens):
    #idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      #get the predictions
      #cropping idx to the last block size tokens.
      idx_cond = idx[:, -block_size:] #idx_cond is the last block size of the input sequence
      logits,loss = self(idx_cond)
      # print(logits.shape)
      # print(logits)
      #focus only on the last time step
      logits = logits[:,-1,:] #becomes (B,C)  all pages, last row of each page, all columns
      # print(logits)
      # print("-----------------------------")
      #apply softmax to get probabilties
      probs = F.softmax(logits,dim =-1) #(B,C) this basically convert the raw data into probabilities or
      # what are the likelihood of occurance of each char

      # print(probs)
      # print("-----------------------------")
      #sample from the distribution
      idx_next= torch.multinomial(probs,num_samples=1) #(B,1)

      # print("-----------------------------")
      # append sampled index to the running sequence

      idx = torch.cat((idx,idx_next),dim=1) #(B,T+1)
      # print(idx)
    return idx
model= BigramLanguageModel()
# logits,loss = m(xb,yb)  #this line automatically calls the forward method bcz forward fn is hardcoded in
# nn module whenever class is called like funtion it automatically call the forward function defined by you

m= model.to(device)

#create Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3) #adamw will change the weights and biases ,
# pramaters will tell which weight and biases need to change ,
# lr(learning_Rate tell by how many step it needs to be changed )
# according to output

for iter in range(max_iters):

    #every once in a while evaluate the loss on train and va l sets
    if iter %eval_interval ==0:
        # previuosly we were using the more or less lucky to get the batch of data
        losses = estimate_loss()
        print(f"iter={iter} train_loss={losses['train_loss']} val_loss={losses['val_loss']}")

    ##sample a batch of data
    xb,yb = get_batch('train')


    #evaluate the loss
    logits,loss = model(xb,yb)  #evalutaing the loss 
    optimizer.zero_grad(set_to_none=True) #zeroing out all the gradients from the previous step or resetting to begin from initial
    # gradient is actual differential to get the minima of the loss
    loss.backward() #getting the gradients fo all the parameters or bckpropogation to get where is the fault that an to minima of loss 
    optimizer.step() #using the gradients to update the parameters according to step size defined.

#generate a sequence of text
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,500)[0].tolist()))

