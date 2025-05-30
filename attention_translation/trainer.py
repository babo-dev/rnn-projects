import time
import os
from utils import timeSince, showPlot
from data.dataset import get_dataloader
from models.encoder import EncoderRNN
from models.seq2seq import AttnDecoderRNN

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_models(encoder, decoder, save_path):
    os.makedirs(save_path, exist_ok=True)

    encoder_save_path = os.path.join(save_path, "encoder.pth")
    torch.save(encoder.state_dict(), encoder_save_path)
    print(f"Encoder saved to {encoder_save_path}")

    decoder_save_path = os.path.join(save_path, "decoder.pth")
    torch.save(decoder.state_dict(), decoder_save_path)
    print(f"Decoder saved to {decoder_save_path}")


def load_models(encoder, decoder, save_path):
    encoder_load_path = os.path.join(save_path, "encoder.pth")
    encoder.load_state_dict(torch.load(encoder_load_path))
    print(f"Encoder loaded from {encoder_load_path}")

    decoder_load_path = os.path.join(save_path, "decoder.pth")
    decoder.load_state_dict(torch.load(decoder_load_path))
    print(f"Decoder loaded from {decoder_load_path}")


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, checkpoint_path, n_epochs, learning_rate=0.001,
          print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    save_models(encoder, decoder, checkpoint_path)
    showPlot(plot_losses)


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 32
    file_path = "data/processed/tuk_processed.txt"
    checkpoint_path = "saved/checkpoints"
    continue_training = True

    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, file_path)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    if continue_training:
        load_models(encoder, decoder, checkpoint_path)

    train(train_dataloader, encoder, decoder, checkpoint_path, 20, print_every=1, plot_every=1)
