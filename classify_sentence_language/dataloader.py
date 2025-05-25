import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import glob


class SentenceDataset(data.Dataset):
    def __init__(self, data_path: str, embedding_model, debugging: bool = False):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.sentences = []
        self.labels = []
        self.label2id = {}
        self.debugging = debugging
        self.init()

    def init(self):
        files = glob.glob(f"{self.data_path}*.txt")
        label_set = set()
        for file in files:
            label = file.split('/')[-1].split('.')[0]
            label_set.add(label)

        self.label2id = {label: idx for idx, label in enumerate(sorted(label_set))}

        for file in files:
            label = file.split('/')[-1].split('.')[0]
            with open(file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if self.debugging and idx > 1000:
                        break
                    sentence = line.strip()
                    encoded_sentence = self.embedding_model.encode(sentence)
                    self.sentences.append(encoded_sentence)
                    self.labels.append(self.label2id[label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]

        sentence_tensor = torch.tensor(sentence, dtype=torch.long)
        return sentence_tensor, label


def add_padding(batch):
    sentences, labels = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    return padded_sentences, torch.tensor(labels, dtype=torch.long)


def train_loader(data_path: str, embedding_model, debugging: bool = False):
    sentence_dataset = SentenceDataset(data_path, embedding_model, debugging)

    loader = data.DataLoader(sentence_dataset, batch_size=32, shuffle=True, collate_fn=add_padding)

    return loader
