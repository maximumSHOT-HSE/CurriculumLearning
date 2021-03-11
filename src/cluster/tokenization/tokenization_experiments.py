from argparse import ArgumentParser
from datasets import load_from_disk

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--dataset', type=str,
                        default='/home/aomelchenko/Bachelor-s-Degree/src/cluster/fine_tuning/sentiment140/sentiment140_init')
    parser.add_argument('--output', type=str,
                        default='/home/aomelchenko/Bachelor-s-Degree/src/cluster/tokenization/sentiment140')

    return parser.parse_args()


def train_tokenizer(dataset, vocab_size):
    trainer = UnigramTrainer(vocab_size=vocab_size)
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(dataset['text'], trainer)

    return tokenizer


def tokenize_dataset(dataset, tokenizer):
    def mapper(x, max_len=512):
        encodings = tokenizer.encode(x['text'])
        input_ids = encodings.ids[:max_len]
        attention_mask = encodings.attention_mask[:max_len]
        token_type_ids = encodings.type_ids[:max_len]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    return dataset.map(mapper)


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_from_disk(args.dataset)

    tokenizer = train_tokenizer(dataset['train'], vocab_size=args.vocab_size)
    tokenized_dataset = tokenize_dataset(dataset['train'], tokenizer)

    tokenized_dataset.save_to_disk(f'{args.output}/{args.vocab_size}')
