import os 
import torch
import tiktoken
import urllib.request
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




"""
Downloads the data sample 'the-veredict', use as a small training dataset 
"""
def download_text_sample():
  if not os.path.exists("the-veredict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


"""
  train_test_split
    Split the text into a trining and validation set.
"""
def train_test_split(text_data, train_ratio=0.9):
  split_idx = int(train_ratio * len(text_data))
  train_data = text_data[:split_idx]
  val_data = text_data[split_idx:]
  return train_data, val_data



"""
Dataset - Creates the dataset to work with based on the text, creates the inputs and targes.   
  txt -> Complete text to work with 
  tokenizer -> the one use to tokenize the text 
  max_length -> The max size of tokens use to make a 'sample'
  stride -> the number of tokens to shift from the previews to take the next sample 
"""
class GPTDataset(Dataset): 
  def __init__(self, txt, tokenizer, max_length, stride):
    # Array of 'samples' created from the original text being the target, the word on the right to the current input.
    self.input_ids = []
    self.target_ids = []

    # Once the text is encoded, the chunck of data samples can be created
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

    # Encode the whole text chunck by chunk, giving jumps of stride size tokens
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i: i+max_length]
      target_chunk = token_ids[i+1: i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  # Number of training examples of the current text
  def __len__(self):
    return len(self.input_ids)

  # The requestes chunk from the text both inputs and targets
  def __getitem__(self, idx): 
    return self.input_ids[idx], self.target_ids[idx]




"""
  This funciton takes a text and creates a dataloader with the specified paramethers. 
  The Tokenizer is tiktoken, used in GPT2
"""
def create_data_loader(txt, batch_size=6, max_length=256, stride=256, shuffle=True, drop_last=True, num_workers=0):
  # Tokenizer use on GPT2
  tokenizer = tiktoken.get_encoding("gpt2")

  # From text to dataloader
  dataset = GPTDataset(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )

  return dataloader




"""
  text_generation
    From a initial text, the model predicts as many 'num_token_generation' tokens, appending themo to the initial text.
"""

def text_generation(model, idx, num_token_generation, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(num_token_generation):            
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:                
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:                  
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:    
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:              
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx



#  ===== Text Manipulation =====
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


# ================================================== Training ==================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = text_generation(
            model=model, 
            idx=encoded,
            num_token_generation=10, 
            context_size=context_size,
            top_k=25,
            temperature=1.4
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("Text Generation Sample")
    print(decoded_text.replace("\n", " "))  
    model.train()