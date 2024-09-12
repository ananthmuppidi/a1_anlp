import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from tqdm import tqdm
import re  

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    
    return text


corpus = load_data('/kaggle/input/auguste-marquet/Auguste_Maquet.txt')


tokens = word_tokenize(corpus)


model_w2v = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
wv = model_w2v.wv


vocab = list(wv.index_to_key)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}


context_size = 5
embed_size = 100
hidden_size = 300
num_epochs = 100
batch_size = 8192
learning_rate = 0.01


data = []
for i in range(context_size, len(tokens)):
    context = [word_to_ix[tokens[j]] for j in range(i-context_size, i)]
    target = word_to_ix[tokens[i]]
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


class LSTMNeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(LSTMNeuralNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden):
        embeds = self.embeddings(inputs)
        lstm_out, hidden = self.lstm(embeds, hidden)
        output = self.fc(lstm_out[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size):

        return (torch.zeros(1, batch_size, hidden_size).to(device),
                torch.zeros(1, batch_size, hidden_size).to(device))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LSTMNeuralNet(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def calculate_perplexity(loss):
    return torch.exp(loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LSTMNeuralNet(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def calculate_perplexity(loss):
    return torch.exp(loss)


train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []


for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0


    train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for context, target in train_loop:
        context, target = context.to(device), target.to(device)


        batch_size = context.size(0)
        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()


        logits, hidden = model(context, hidden)


        hidden = (hidden[0].detach(), hidden[1].detach())


        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_perplexity = calculate_perplexity(torch.tensor(avg_train_loss)).item()


    train_losses.append(avg_train_loss)
    train_perplexities.append(train_perplexity)


    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
        for context, target in val_loop:
            context, target = context.to(device), target.to(device)


            batch_size = context.size(0)
            hidden = model.init_hidden(batch_size)

            logits, hidden = model(context, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(logits, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_perplexity = calculate_perplexity(torch.tensor(avg_val_loss)).item()


    val_losses.append(avg_val_loss)
    val_perplexities.append(val_perplexity)

    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, '
          f'Train Perplexity = {train_perplexity:.4f}, Val Perplexity = {val_perplexity:.4f}')


import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Epochs')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(train_perplexities, label='Train Perplexity')
plt.plot(val_perplexities, label='Validation Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Train and Validation Perplexity Over Epochs')
plt.legend()
plt.show()

def create_perplexity_file(dataloader, model, vocab, filename):
    model.eval()
    perplexities = []
    
    with torch.no_grad():
        with open(filename, 'w') as f:
            for context, target in dataloader:
                context, target = context.to(device), target.to(device)

                batch_size = context.size(0)
                hidden = model.init_hidden(batch_size)

                logits, hidden = model(context, hidden)
                loss = criterion(logits, target)
                perplexity = calculate_perplexity(loss).item()
                perplexities.append(perplexity)

                for sentence in context:
                    first_5_words = ' '.join([vocab[idx.item()] for idx in sentence[:5]])
                    f.write(f"{first_5_words}\t{perplexity:.4f}\n")

            # Calculate and write the average perplexity at the end of the file
            avg_perplexity = sum(perplexities) / len(perplexities)
            f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")


train_dataloader_single = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
val_dataloader_single = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
test_dataloader_single = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

create_perplexity_file(train_dataloader_single, model, ix_to_word, "2021101122-LM2-train-perplexity.txt")
create_perplexity_file(val_dataloader_single, model, ix_to_word, "2021101122-LM2-val-perplexity.txt")
create_perplexity_file(test_dataloader_single, model, ix_to_word, "2021101122-LM2-test-perplexity.txt")

import torch

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model(model, path="lstm_model.pth")
