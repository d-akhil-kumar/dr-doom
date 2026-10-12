import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text  = (
"Once upon a time, in a land far, far away, there lived a wise old owl. |<endoftext>| This owl had seen many things in its long life and had learned the secrets of the forest. |<endoftext>| One day, a young rabbit approached the owl, seeking advice on how to find food during the harsh winter months. |<endoftext>| The owl shared its wisdom, teaching the rabbit how to forage for hidden berries and nuts, and how to store them for later use. |<endoftext>| The rabbit thanked the owl and promised to share this knowledge with others in the forest. |<endoftext>| From that day on, the forest thrived, and all the animals lived in harmony, thanks to the wise old owl's guidance. |<endoftext>|"
)


ids = tokenizer.encode(text, allowed_special={"|<endoftext>|"})

print("Encoded IDs:", ids)


decoded_text = tokenizer.decode(ids)
print("Decoded Text:", decoded_text)