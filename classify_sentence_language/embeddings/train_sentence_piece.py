import argparse
import glob

import sentencepiece as spm


def train_sentence_piece(train_dir: str, vocab_size: int = 3000, output_dir: str = 'embeddings/embedding'):
    if not train_dir.endswith('/'):
        train_dir += '/'

    files = ','.join(glob.glob(f"{train_dir}*.txt"))
    arg = f'--input={files} --model_prefix={output_dir} --vocab_size={vocab_size}'
    spm.SentencePieceTrainer.Train(arg)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train SentencePiece model')
    argparser.add_argument('--train_dir', type=str, required=True, help='Directory containing training text files')
    argparser.add_argument('--vocab_size', type=int, default=2000, help='Vocabulary size')
    argparser.add_argument('--output', type=str, default='embeddings/embedding', help='Output model name')
    args = argparser.parse_args()

    train_sentence_piece(args.train_dir, args.vocab_size, args.output)


