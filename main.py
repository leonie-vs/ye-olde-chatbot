from chatbot import Chatbot

bot = Chatbot()

encoded = bot.encode_prompt("Hello, how are you?")
print(encoded)

decoded_prompt = bot.decode_reply([15496, 11, 703, 389, 345, 30]) # Pass in list of token IDs generated from tokenizer
print(decoded_prompt)

prompt = "What is the weather like today?"
reply = bot.generate_reply(prompt)
print(f"Prompt: {prompt}")
print(f"Reply: {reply}")