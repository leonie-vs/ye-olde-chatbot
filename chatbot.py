from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.chat_history_ids = None
        self.system_prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n"
    
    def reset_history(self):
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
    
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
        
        model_input = model_input.to(self.device)
            
        output = self.model.generate( # generate reply 
            model_input,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.8, 
            top_p=0.9, 
            top_k=40
        )

        reply_start = model_input.shape[-1] 
        reply_token_ids = output[:, reply_start:]  # get reply_token_ids

        self.chat_history_ids = torch.cat([model_input, reply_token_ids], dim=1) # update chat_history_ids
        
        return self.decode_reply(reply_token_ids[0]) # return decoded reply 
        


    



