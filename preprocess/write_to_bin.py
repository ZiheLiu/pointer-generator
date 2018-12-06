import os
import struct

from tensorflow.core.example import example_pb2

from preprocess import file_utils, constants


def write_to_bin(source_filename, target_filename, output_filename):
    source_sentences = file_utils.read_file_to_string(source_filename).split('\n')
    target_sentences = file_utils.read_file_to_string(target_filename).split('\n')

    with open(output_filename, 'wb') as fout:
        for source_sentence, target_sentence in zip(source_sentences, target_sentences):
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([bytes(source_sentence, 'utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([bytes('<s>{}</s>'.format(target_sentence), 'utf-8')])

            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            fout.write(struct.pack('q', str_len))
            fout.write(struct.pack('%ds' % str_len, tf_example_str))


def merge_vocab(source_vocab_filename, target_vocab_filename, merged_vocab_filename):
    source_vocab = file_utils.read_file_to_string(source_vocab_filename).split('\n')
    target_vocab = file_utils.read_file_to_string(target_vocab_filename).split('\n')

    merged_word_set = dict()
    for word in source_vocab:
        merged_word_set[word] = True
    for word in target_vocab:
        merged_word_set[word] = True

    merged_words_with_freq = ['{} tmp'.format(word) for word in merged_word_set.keys()]
    file_utils.write_string_to_file(merged_vocab_filename, '\n'.join(merged_words_with_freq))


def main():
    write_to_bin(os.path.join(constants.RAW_DIR, 'lower_raw_data.train.source'),
                 os.path.join(constants.RAW_DIR, 'lower_raw_data.train.target'),
                 os.path.join(constants.INPUT_DIR, 'lower_raw_data.train_000.bin'))

    write_to_bin(os.path.join(constants.RAW_DIR, 'lower_raw_data.eval.source'),
                 os.path.join(constants.RAW_DIR, 'lower_raw_data.eval.target'),
                 os.path.join(constants.INPUT_DIR, 'lower_raw_data.eval_000.bin'))

    write_to_bin(os.path.join(constants.RAW_DIR, 'lower_raw_data.test.source'),
                 os.path.join(constants.RAW_DIR, 'lower_raw_data.test.target'),
                 os.path.join(constants.INPUT_DIR, 'lower_raw_data.test_000.bin'))

    merge_vocab(os.path.join(constants.RAW_DIR, 'lower_raw_data.vocab.source'),
                os.path.join(constants.RAW_DIR, 'lower_raw_data.vocab.target'),
                os.path.join(constants.INPUT_DIR, 'lower_raw_data.vocab'))


if __name__ == '__main__':
    main()
