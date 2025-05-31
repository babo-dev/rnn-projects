import time
import os
from utils import timeSince
from config import load_config
from data.dataset import get_dataloader
from models.encoder import EncoderRNN, Encoder
from models.decoder import Decoder
from models.attentions import DotProductAttention
from models.seq2seq import AttnDecoderRNN, Seq2SeqDotAttention

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

    # save_models(encoder, decoder, checkpoint_path)
    # showPlot(plot_losses)


def run_tutorial(hidden_size, batch_size, file_path, checkpoint_path, continue_training=True):
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, file_path)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    if continue_training:
        load_models(encoder, decoder, checkpoint_path)

    train(train_dataloader, encoder, decoder, checkpoint_path, 5, print_every=1)


def train_universal(train_dataloader, model, checkpoint_path, n_epochs=30, learning_rate=0.001, print_every=1):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=learning_rate)
    optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for data in train_dataloader:
            input_tensor, target_tensor = data

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            outputs, _, _ = model(input_tensor, target_tensor)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_tensor.view(-1))
            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder.step()

            total_loss += loss.item()

        print_loss_total += total_loss / len(train_dataloader)
        plot_loss_total += total_loss / len(train_dataloader)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

    save_models(model.encoder, model.decoder, checkpoint_path)


def run_universal(hidden_size,  batch_size, file_path, checkpoint_path, continue_training=False):
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, file_path)

    attn = DotProductAttention()
    encoder = Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_lang.n_words, attn).to(device)
    model = Seq2SeqDotAttention(encoder, decoder).to(device)

    if continue_training:
        load_models(encoder, decoder, checkpoint_path)

    train_universal(train_dataloader, model, checkpoint_path, 5)


if __name__ == "__main__":
    config_path = "config/config.yaml"
    cfg = load_config(config_path)
    checkpoint_path = "saved/checkpoints"
    continue_train = False

    run_universal(cfg.model.hidden_dim, cfg.training.batch_size, cfg.data.train_path, checkpoint_path, continue_train)
    # run_tutorial(cfg.model.hidden_dim, cfg.training.batch_size, cfg.data.train_path, checkpoint_path, continue_train)
