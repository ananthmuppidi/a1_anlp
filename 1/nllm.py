import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text.lower()

corpus = load_data('/kaggle/input/auguste-marquet/Auguste_Maquet.txt')
tokens = word_tokenize(corpus)  # Tokenization

model_w2v = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
wv = model_w2v.wv

vocab = list(wv.index_to_key)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}



context_size = 5  
embed_size = 100  
hidden_size = 300 
num_epochs = 50
batch_size = 8192
learning_rate = 0.01


data = []
for i in range(context_size, len(tokens)):
    context = [word_to_ix[tokens[j]] for j in range(i-context_size, i)]  # 5 previous words
    target = word_to_ix[tokens[i]]  # Target is the current word (ith word)
    data.append((context, target))


train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch):
    context, target = zip(*batch)
    context = torch.tensor(context)
    target = torch.tensor(target)
    return context, target


train_dataset = WordDataset(train_data)
val_dataset = WordDataset(val_data)
test_dataset = WordDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)



class NeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_rate):
        super(NeuralNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(context_size * embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.size(0), -1)  # Flatten the embeddings
        out = self.dropout(torch.relu(self.linear1(embeds)))
        out = self.linear2(out)  
        return out


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def calculate_perplexity(loss):
    return torch.exp(loss)

def train_and_evaluate_model(hidden_size, dropout_rate, optimizer_type):
    model = NeuralNet(vocab_size, embed_size, hidden_size, dropout_rate).to(device)
    
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for context, target in train_dataloader:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_perplexity = calculate_perplexity(torch.tensor(avg_train_loss)).item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_dataloader:
                context, target = context.to(device), target.to(device)
                logits = model(context)
                loss = criterion(logits, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_perplexity = calculate_perplexity(torch.tensor(avg_val_loss)).item()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Perplexity: {train_perplexity:.4f} | Val Perplexity: {val_perplexity:.4f}")

    return model, train_losses, val_losses, train_perplexities, val_perplexities

hidden_sizes = [200, 300]
dropout_rates = [0.1, 0.3]
optimizers = ["SGD"]

all_results = []
best_val_loss = float('inf')
best_hyperparameters = None
best_train_losses = None
best_val_losses = None
best_train_perplexities = None
best_val_perplexities = None
best_model = None

total_combinations = len(hidden_sizes) * len(dropout_rates) * len(optimizers)

with tqdm(total=total_combinations, desc="Hyperparameter Tuning") as hp_bar:
    for hidden_size, dropout_rate, optimizer in itertools.product(hidden_sizes, dropout_rates, optimizers):
        print(f"\nTraining with hidden_size={hidden_size}, dropout_rate={dropout_rate}, optimizer={optimizer}\n")
        
        model, train_losses, val_losses, train_perplexities, val_perplexities = train_and_evaluate_model(hidden_size, dropout_rate, optimizer)
        
        final_val_loss = val_losses[-1]
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_hyperparameters = (hidden_size, dropout_rate, optimizer)
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_perplexities = train_perplexities
            best_val_perplexities = val_perplexities
            best_model = model
        
        all_results.append({
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities
        })
        
        hp_bar.update(1)

print(f"Best hyperparameters: Hidden Size = {best_hyperparameters[0]}, Dropout Rate = {best_hyperparameters[1]}, Optimizer = {best_hyperparameters[2]}")
print(f"Best Train Loss: {best_train_losses[-1]:.4f} | Best Val Loss: {best_val_losses[-1]:.4f}")
print(f"Best Train Perplexity: {best_train_perplexities[-1]:.4f} | Best Val Perplexity: {best_val_perplexities[-1]:.4f}")


import matplotlib.pyplot as plt
import numpy as np

# Plot Losses for Train and Validation Sets for all hyperparameters
def plot_loss(all_results):
    plt.figure(figsize=(10, 6))
    for result in all_results:
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        hyperparams = f"HS={result['hidden_size']}, DR={result['dropout_rate']}, Opt={result['optimizer']}"
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f'Train Loss ({hyperparams})')
        plt.plot(epochs, val_losses, label=f'Val Loss ({hyperparams})', linestyle='dashed')
    
    plt.title('Train and Validation Loss for Different Hyperparameters')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot Perplexity for Train and Validation Sets for all hyperparameters
def plot_perplexity(all_results):
    plt.figure(figsize=(10, 6))
    for result in all_results:
        train_perplexities = result['train_perplexities']
        val_perplexities = result['val_perplexities']
        hyperparams = f"HS={result['hidden_size']}, DR={result['dropout_rate']}, Opt={result['optimizer']}"
        epochs = np.arange(1, len(train_perplexities) + 1)
        plt.plot(epochs, train_perplexities, label=f'Train Perplexity ({hyperparams})')
        plt.plot(epochs, val_perplexities, label=f'Val Perplexity ({hyperparams})', linestyle='dashed')
    
    plt.title('Train and Validation Perplexity for Different Hyperparameters')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()

# Plot the average loss across all hyperparameter configurations
def plot_avg_loss(all_results):
    avg_train_losses = np.mean([result['train_losses'] for result in all_results], axis=0)
    avg_val_losses = np.mean([result['val_losses'] for result in all_results], axis=0)
    epochs = np.arange(1, len(avg_train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_losses, label='Average Train Loss')
    plt.plot(epochs, avg_val_losses, label='Average Val Loss', linestyle='dashed')
    
    plt.title('Average Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the average perplexity across all hyperparameter configurations
def plot_avg_perplexity(all_results):
    avg_train_perplexities = np.mean([result['train_perplexities'] for result in all_results], axis=0)
    avg_val_perplexities = np.mean([result['val_perplexities'] for result in all_results], axis=0)
    epochs = np.arange(1, len(avg_train_perplexities) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_perplexities, label='Average Train Perplexity')
    plt.plot(epochs, avg_val_perplexities, label='Average Val Perplexity', linestyle='dashed')
    
    plt.title('Average Train and Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()

# Call the plotting functions
plot_loss(all_results)
plot_perplexity(all_results)
plot_avg_loss(all_results)
plot_avg_perplexity(all_results)

# Create dataloaders with batch size 1 for calculating perplexity
train_dataloader_single = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
val_dataloader_single = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
test_dataloader_single = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

def create_perplexity_file(dataloader, model, vocab, filename):
    model.eval()
    with torch.no_grad():
        with open(filename, 'w') as f:
            for context, target in dataloader:
                context, target = context.to(device), target.to(device)
                logits = model(context)
                loss = nn.CrossEntropyLoss()(logits, target)
                perplexity = calculate_perplexity(loss).item()
                

                for sentence in context:
                    # Convert the first 5 words in the context back to words from their indices
                    first_5_words = ' '.join([vocab[idx.item()] for idx in sentence[:5]])
                    f.write(f"{first_5_words}\t{perplexity:.4f}\n")
                    
                    

# Create perplexity files for train, validation, and test sets
create_perplexity_file(train_dataloader_single, best_model, vocab, "train_perplexity.txt")
create_perplexity_file(val_dataloader_single, best_model, vocab, "val_perplexity.txt")
create_perplexity_file(test_dataloader_single, best_model, vocab, "test_perplexity.txt")

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model(model, path="nllm_model.pth")

