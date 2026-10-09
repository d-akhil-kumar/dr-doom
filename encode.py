import re

class EncodeV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: word for word, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token for token in preprocessed if token.strip() != '']
        preprocessed = [token if token in self.str_to_int else "<|unk|>" for token in preprocessed]

        ids = [self.str_to_int[token] for token in preprocessed if token in self.str_to_int]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)  # Remove space before punctuation
        return text