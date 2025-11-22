from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.chat_history_ids = None
        self.system_prompt = "You are a helpful assistant. Respond to the end of this conversation accordingly.\n"
    
    def reset_history(self):
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        
        prompt = prompt + "\n" # append newline character
        input_ids = self.encode_prompt(prompt) # encode prompt to get input ids
        
        if self.chat_history_ids is None: # check if this is the first prompt
            sys_ids = self.encode_prompt(self.system_prompt) # if yes, get system prompt ids 
            model_input = torch.cat([sys_ids, input_ids], dim=1) # pass prompt combined with system_prompt
        else: # if not first, pass prompt combined with chat_history_ids
            model_input = torch.cat([self.chat_history_ids, input_ids], dim=1)
            
        output = self.model.generate( # generate reply 
            model_input,
            max_new_tokens=50,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.9, 
            top_p=0.8, 
            top_k=50
        )

        reply_start = model_input.shape[-1] 
        reply_token_ids = output[:, reply_start:]  # get reply_token_ids

        self.chat_history_ids = torch.cat([model_input, reply_token_ids], dim=1) # update chat_history_ids
        
        return self.decode_reply(reply_token_ids[0]) # return decoded reply 
        


    



