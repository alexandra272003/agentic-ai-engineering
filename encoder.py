
class MyTokenizer:

    def __init__(self):
        # Custom vocabulary
        self.vocab = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
            "i": 9,
            "j": 10,
            "k": 11,
            "l": 12,
            "m": 13,
            "n": 14,
            "o": 15,
            "p": 16,
            "q": 17,
            "r": 18,
            "s": 19,
            "t": 20,
            "u": 21,
            "v": 22,
            "w": 23,
            "x": 24,
            "y": 25,
            "z": 26,
            " ": 27
        }

        # Reverse vocabulary for decoding
        self.reverse_vocab = {
            value: key for key, value in self.vocab.items()
        }

    def encoder(self, message):
        tokens = []

        for char in message.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                raise ValueError(f"Unknown character: {char}")

        return tokens

    def decoder(self, tokens):
        message = ""

        for token in tokens:
            if token in self.reverse_vocab:
                message += self.reverse_vocab[token]
            else:
                raise ValueError(f"Unknown token ID: {token}")

        return message


# Testing
tokenizer = MyTokenizer()

msg = "hello"

encoded = tokenizer.encoder(msg)
decoded = tokenizer.decoder(encoded)

print("Original Message:", msg)
print("Encoded Tokens:", encoded)
print("Decoded Message:", decoded)
