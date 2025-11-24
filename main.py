from chatbot import Chatbot

bot = Chatbot()

#bot.reset_history()
#print(bot.chat_history_ids)

print("Hello, welcome to my Chatbot!\n")
print("Please type your message below to start a conversation. Type 'quit' to exit.\n")

while True:

    prompt = input("You: ")
    reply = bot.generate_reply(prompt)

    if prompt == "quit":
        print("\nGoodbye!")
        break

    print(f"Bot: {reply}\n")

