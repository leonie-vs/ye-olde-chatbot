from chatbot import Chatbot
import torch

def test_initial_history_is_none():
    bot = Chatbot()
    assert bot.chat_history_ids is None

def test_reset_history_converts_it_back_to_none():
    bot = Chatbot()
    bot.generate_reply("Hello")
    assert bot.chat_history_ids is not None
    bot.reset_history()
    assert bot.chat_history_ids is None

def test_first_message_includes_system_prompt_and_first_prompt():
    bot = Chatbot()
    bot.generate_reply("Hello")
    history_text = bot.tokenizer.decode(bot.chat_history_ids[0])
    assert "You are a helpful assistant. Respond to the end of this conversation accordingly.\n" in history_text
    assert "Hello" in history_text

def test_history_gets_larger_after_each_prompt():
    bot = Chatbot()
    bot.generate_reply("Hello")
    first_len = bot.chat_history_ids.shape[-1]
    bot.generate_reply("How are you?")
    second_len = bot.chat_history_ids.shape[-1]
    assert second_len > first_len

def test_reply_is_string():
    bot = Chatbot()
    reply = bot.generate_reply("Hello")
    assert isinstance(reply, str)
    assert len(reply) > 0