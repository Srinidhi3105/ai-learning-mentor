import tiktoken


"""• Definition: A token is a small piece of text.
 A tokenizer makes these tokens before a Large Language Model (LLM) reads the text."""

encoding = tiktoken.get_encoding("o200k_base")

text = "I am learning llm engieering "

tokens = encoding.encode(text)

print("Original text:", text)

print("\nNumber of tokens:", len(tokens))

print("\nTokens IDs:")
print(len(tokens))

print("\nDecoded tokens:")

for token in tokens:
    print(token, "->", encoding.decode([token]))
