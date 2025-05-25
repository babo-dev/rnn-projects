from dataloader import train_loader
from models import RawRNNModel, LSTMModel
import sentencepiece as spm
import torch


def train(data_path: str, embedding_model_path: str, num_epochs: int, save_path: str, debugging: bool = False, device: torch.device = torch.cpu):
    sp = spm.SentencePieceProcessor()
    sp.Load(embedding_model_path)

    dataloader = train_loader(data_path, embedding_model=sp, debugging=debugging)

    vocab_size = sp.GetPieceSize()
    embedding_dim = 64
    hidden_dim = 64
    output_dim = len(dataloader.dataset.label2id)
    num_layers = 2

    # model = RawRNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout=0.2)
    print(model)

    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (sentences, labels) in enumerate(dataloader):
            sentences = sentences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), save_path)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for sentences, labels in dataloader:
            sentences = sentences.to(device)
            labels = labels.to(device)

            outputs = model(sentences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)

    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    data_path = 'data/train/'
    save_dir = "data/models/model.pth"
    emb_path = 'embeddings/embedding.model'
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(data_path, emb_path, num_epochs, save_dir, debugging=True, device=device)

