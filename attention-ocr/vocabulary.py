import numpy as np


class Vocabulary:
    pad = '<PAD>'  # padding
    start = '<START>'  # start of sentence
    eos = '<END>'  # end of sentence

    def __init__(self, letters=" QWERTYUIOPASDFGHJKLZXCVBNM'-:1234567890+()", max_txt_length=33):
        self.vocabulary = [char for char in letters]
        self.vocabulary += ['<END>', '<START>', '<PAD>']
        self.max_txt_length = max_txt_length

    def text_to_labels(self, text):
        label = list(map(lambda x: self.vocabulary.index(x), text))
        label = [self.vocabulary.index('<START>')] + label + [self.vocabulary.index('<END>')]
        label += [self.vocabulary.index('<PAD>') for _ in range(self.max_txt_length - len(label))]  # add padding

        return label

    def labels_to_text(self, labels):
        return ''.join(list(map(lambda x: self.vocabulary[int(x)] if x < len(self.vocabulary) else "", labels)))

    def word_index(self, word):
        return self.vocabulary.index(word)
