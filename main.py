import re
from encode import EncodeV1



with open("./pretraining-data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Length of text: ", len(raw_text))
# print(raw_text[:500])  # Print the first 500 characters of the text 



preprcessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprcessed = [token for token in preprcessed if token.strip() != '']  # Remove empty tokens

# print(preprcessed[:100])  # Print the first 100 tokens

all_words = sorted(set(preprcessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])  # Add special tokens for padding and unknown words
vocab_size = len(all_words)

# print("Vocabulary: ", all_words)
print("Vocabulary size: ", vocab_size)


vocab = {word: i for i, word in enumerate(all_words)} 

# print("Vocabulary dictionary: ", vocab)

encode = EncodeV1(vocab)

text= """"It's the last he painted, you know,"
          Mrs. Gisburn said with pardonable pride."""

ids = encode.encode(text)
print("Encoded IDs: ", ids)

decoded_text = encode.decode(ids)
print("Decoded text: ", decoded_text)



text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace"
text_combined = text1 + " <|endoftext|> " + text2

ids_combined = encode.encode(text_combined)
print("Encoded IDs for combined text: ", ids_combined)

decoded_combined_text = encode.decode(ids_combined)
print("Decoded combined text: ", decoded_combined_text)