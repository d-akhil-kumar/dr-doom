import tiktoken

with open("./pretraining-data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(raw_text)

print("Number of tokens:", len(tokens))

context_size = 4

x = tokens[:context_size]
y = tokens[1:context_size+1]

print("Input tokens (x):", x)
print("Target tokens (y):", y)

for i in range(1,context_size + 1):
    input = x[:i]
    output = y[i - 1]

    # print(input, "->", output)
    # print(tokenizer.decode(input), "->", tokenizer.decode([output]))

# decoded_text = tokenizer.decode(tokens)
# print("Decoded Text:", decoded_text)