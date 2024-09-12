# importing libraries
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , TensorDataset
from collections import Counter
import numpy as np
import torch.nn as nn
from gensim.models import Word2Vec
import re
from tqdm import tqdm

data_path = '/kaggle/input/auguste-maquet-txt/Auguste_Maquet.txt'
with open(data_path, 'r') as file:
    corpus = file.read()
corpus = corpus.strip()
sentences = nltk.sent_tokenize(corpus)
cleaned_sentences = [re.sub(r'[^A-Za-z0-9\s]', '', sentence) for sentence in sentences]
# cleaned_sentences = [sent for sent in sentences if len(word_tokenize(sent)) < 20 and len(word_tokenize(sent)) >= 5]
max_seq_length = 0
for sent in cleaned_sentences:
    max_seq_length = max(max_seq_length, len(word_tokenize(sent)))
print(f"Maximum sequence length: {max_seq_length}")

num_train = int(0.7 * len(cleaned_sentences))
num_val = int(0.9 * len(cleaned_sentences))

train = cleaned_sentences[:num_train]
val = cleaned_sentences[num_train:num_val]
test = cleaned_sentences[num_val:]

print(f"Train, Val, Test splits : {len(train)}, {len(val)}, {len(test)}")

sequence_length = 21
start_token = '<SOS>'
end_token = '<EOS>'
pad_token = '<PAD>'
unknown_token = '<UNK>'
split_sentences = []
for sentence in cleaned_sentences:
    words = nltk.word_tokenize(sentence)
    split_sentences.append(words)
split_sentences.append([unknown_token, pad_token])
def create_dataset(sentences):
    
    X = []
    y = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence, language='english')
        words = [word for word in words if word.isalpha()]

        if len(words) < sequence_length:
            words = words + [pad_token] * (sequence_length - len(words))
        
        context = words[:sequence_length - 1]
        target  = words[1:sequence_length]

        X.append(context)
        y.append(target)

    return X, y


X_train, y_train = create_dataset(train)
X_val, y_val = create_dataset(val)
X_test, y_test = create_dataset(test)

print('Number of training samples:', len(X_train))
print('Number of validation samples:', len(X_val))
print('Number of test samples:', len(X_test))
from gensim.models import Word2Vec
word2vec = Word2Vec(sentences=split_sentences, vector_size=100, window=sequence_length, sg=0, min_count=1)

print('Vocabulary size:', len(word2vec.wv))
def w2v(words, word2vec):
    vectors = []
    for word in words:
        if word in word2vec.wv:
            vectors.append(word2vec.wv.key_to_index[word])
        else:
            vectors.append(word2vec.wv.key_to_index['UNK'])
    return vectors



X_train = torch.tensor([w2v(words, word2vec) for words in X_train])
X_val = torch.tensor([w2v(words, word2vec) for words in X_val])
X_test = torch.tensor([w2v(words, word2vec) for words in X_test])
y_train = torch.tensor([w2v(words, word2vec) for words in y_train])
y_val = torch.tensor([w2v(words, word2vec) for words in y_val])
y_test = torch.tensor([w2v(words, word2vec) for words in y_test])

device = "cuda"
class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, embedding_dim, heads, num_decoder_layers):
        super(DecoderOnly, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=heads)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 20, embedding_dim)) 
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding 
        seq_len = x.size(1) 
        mask = self._mask(seq_len).to(x.device) 
        x = self.transformer_decoder(x.transpose(0, 1), x.transpose(0, 1), tgt_mask=mask) 
        x = self.fc(x) 
        return x

    def _mask(self, seq_len):

        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)  
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

embedding_dim = word2vec.wv.vector_size
heads = 4
layers = 2
vocab_size = len(word2vec.wv)

model = DecoderOnly(vocab_size, embedding_dim, heads, layers).to(device)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 1024         
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_batches = len(train_dataloader)
val_batches = len(val_dataloader)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

from tqdm import tqdm

train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def train(model, train_dataloader, val_dataloader, criterion, optimizer, n_epochs, checkpoint_path=None):
    for epoch in range(n_epochs):
        model.train()  
        curr_train_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs} [Training]', leave=False):
            optimizer.zero_grad()
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            outputs = outputs.view(-1, outputs.size(-1))
            batch_y = batch_y.view(-1)

            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            curr_train_loss += loss.item()
        
        train_loss = curr_train_loss / len(train_dataloader)
        train_perplexity = calculate_perplexity(train_loss).item()
        
        train_losses.append(train_loss)
        train_perplexities.append(train_perplexity)

        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{n_epochs} [Validation]', leave=False):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                outputs = outputs.view(-1, outputs.size(-1))
                batch_y = batch_y.view(-1)

                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_dataloader)
        val_perplexity = calculate_perplexity(val_loss).item()
        
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}')

        if checkpoint_path is not None and val_loss <= min(val_losses):
            torch.save(model.state_dict(), checkpoint_path + f'/transformer.pt')

    return {
        'train_loss': train_losses, 
        'val_loss': val_losses,
        'train_perplexity': train_perplexities,
        'val_perplexity': val_perplexities
    }

n_epochs = 50
losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, n_epochs, checkpoint_path='/kaggle/working')

train_losses = losses['train_loss']
val_losses = losses['val_loss']


plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()

train_perplexities = losses['train_perplexity']
val_perplexities = losses['val_perplexity']
plt.plot(train_perplexities, label='Training Perplexity')
plt.plot(val_perplexities, label='Validation Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.title('Training and Validation Perplexities')
plt.show()
import os

def calculate_and_save_perplexities(dataloader, model, vocab, filename):
    model.eval()
    perplexities = []
    with torch.no_grad():
        with open(filename, 'w') as f:
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                outputs = outputs.view(-1, outputs.size(-1))
                batch_y = batch_y.view(-1)

                loss = criterion(outputs, batch_y)
                perplexity = calculate_perplexity(loss.item()).item()
                perplexities.append(perplexity)

                for sentence in batch_X:
                    # Convert first 5 words from indices to words using the vocabulary
                    first_5_words = ' '.join([vocab.index_to_key[idx.item()] for idx in sentence[:5]])
                    f.write(f"{first_5_words}\t{perplexity:.4f}\n")
            
            # Calculate and write average perplexity at the end of the file
            avg_perplexity = sum(perplexities) / len(perplexities)
            f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")
    print(f"Perplexities saved to {filename}")

# Assuming 'word2vec' and 'split_sentences' have already been trained and split
# Calculate perplexities for train, validation, and test sets
vocab = word2vec.wv  # Extracting the trained word vectors

train_dataloader_single = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_dataloader_single = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader_single = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Save perplexities for train, validation, and test sets
calculate_and_save_perplexities(train_dataloader_single, model, vocab, '/kaggle/working/train_perplexity.txt')
calculate_and_save_perplexities(val_dataloader_single, model, vocab, '/kaggle/working/val_perplexity.txt')
calculate_and_save_perplexities(test_dataloader_single, model, vocab, '/kaggle/working/test_perplexity.txt')

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model(model, path="transformer_model.pth")
