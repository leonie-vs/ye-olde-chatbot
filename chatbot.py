from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.chat_history_ids = None
    
    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        
        prompt = prompt + "\n" # append newline character
        encoded_prompt = self.encode_prompt(prompt) # encode prompt
        input_ids = encoded_prompt["input_ids"] # get input ids

        output = self.model.generate( # generate reply based on input ids
            input_ids, 
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=20,
            do_sample=True, 
            temperature=0.9, 
            top_p=0.8, 
            top_k=50
            )
        reply_token_ids = output[:, input_ids.shape[-1]:] # get output ids 

        return str(self.decode_reply(reply_token_ids[0])) # decode reply based on output ids


    



