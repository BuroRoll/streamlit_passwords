import streamlit as st
import torch
import torch.nn as nn
import pickle


# Определение классов Generator и Discriminator
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, noise_dim):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + noise_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.noise_dim = noise_dim

    def forward(self, x):
        noise = torch.randn(x.size(0), self.noise_dim).to(x.device)
        embedded = self.embedding(x).unsqueeze(1)
        noise = noise.unsqueeze(1)
        combined = torch.cat((embedded, noise), dim=2)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(combined, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.softmax(out)


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# Загрузка словарей
with open('word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open('index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

# Параметры модели
vocab_size = len(word_to_index)
embedding_dim = 100
hidden_size = 128
noise_dim = 10

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация и загрузка моделей
G = Generator(vocab_size, embedding_dim, hidden_size, noise_dim).to(device)
D = Discriminator(vocab_size, embedding_dim, hidden_size).to(device)

G.load_state_dict(torch.load('generator.pth', map_location=device))
D.load_state_dict(torch.load('discriminator.pth', map_location=device))


# Функции для преобразования слов в индексы и обратно
def word_to_idx(word):
    return word_to_index.get(word, word_to_index['<UNK>'])


def idx_to_word(idx):
    return index_to_word.get(idx, '<UNK>')


# Функция генерации вариантов
def generate_variants(ha, model_G, word_to_index, index_to_word, num_variants=5):
    ha_idx = torch.tensor([word_to_idx(ha)], dtype=torch.long).to(device)
    variants = []
    for _ in range(num_variants):
        with torch.no_grad():
            generated_probs = model_G(ha_idx)
            generated_idx = generated_probs.argmax(dim=1).item()
        generated_word = idx_to_word(generated_idx)
        variants.append(generated_word)
    return variants


# Создание Streamlit приложения
st.title('Passwords Generation')

ha_input = st.text_input('Enter the input text (hash):')
num_variants = st.number_input('Number of variants:', min_value=1, max_value=10, value=5)

if st.button('Generate Variants'):
    if ha_input:
        with st.spinner('Generating...'):
            variants = generate_variants(ha_input, G, word_to_index, index_to_word, num_variants)
        st.success('Variants generated successfully!')
        st.write('Input:', ha_input)
        st.write('Generated Variants:')
        for variant in variants:
            st.write(variant)
    else:
        st.error('Please enter the input text.')
