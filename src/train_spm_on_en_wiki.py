import argparse

import tensorflow_datasets as tfds
import tensorflow as tf
import io
import sentencepiece as spm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Code of the dataset in tfds', default='wiki40b/en')
    parser.add_argument('--vocab-size', type=int, help='The size of vocabulary', default=10000)
    parser.add_argument('--save', type=str, help='Path to the file where spm model will be saved')
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--model-type', type=str, default='bpe')
    parser.add_argument('--dump', type=str, help='Path to the dump directory, where tfds is stored')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # sp = spm.SentencePieceProcessor()
    # sp.load('test.model')
    # print(sp.vocab_size())
    # for i in range(sp.vocab_size()):
    #     print(i, sp.id_to_piece(i))

    dataset, dataset_info = tfds.load(
        name=args.dataset,
        data_dir=args.dump,
        with_info=True,
        split=tfds.Split.TRAIN,
        shuffle_files=False,
        download=args.download
    )

    def article_to_text(text):
        return text.numpy().decode('utf-8')

    dataset_text = dataset.map(
        lambda article: tf.py_function(func=article_to_text, inp=[article['text']], Tout=tf.string)
    )

    model = io.BytesIO()

    spm.SentencePieceTrainer.train(
        sentence_iterator=dataset_text.as_numpy_iterator(),
        model_writer=model,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )

    with open(args.save, 'wb') as f:
        f.write(model.getvalue())
