from chatbot import Chatbot

bot = Chatbot()

encoded = bot.encode_prompt("Hello, how are you?")
print(encoded)

reply = bot.decode_reply([15496, 11, 703, 389, 345, 30]) # Pass in list of token IDs generated from tokenizer
print(reply)