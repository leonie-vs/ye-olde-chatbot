from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)


    



