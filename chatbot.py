from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    
    def __init__(self):

        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.system_prompt_ids = self.tokenizer("<|system|>\nYou are a helpful assistant.<|end|>\n", return_tensors="pt")["input_ids"].to(self.device)
        self.chat_history_ids = None

    def reset_history(self):
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        
        prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"
        input_ids = self.encode_prompt(prompt) # encode prompt to get input ids
        
        if self.chat_history_ids is None: # check if this is the first prompt
            model_input = torch.cat([self.system_prompt_ids, input_ids], dim=1) # pass prompt ids combined with system_prompt ids
        else: # if not first, pass prompt combined with chat_history_ids
            model_input = torch.cat([self.chat_history_ids, input_ids], dim=1)
        
        model_input = model_input.to(self.device)
            
        output = self.model.generate( # generate reply 
            model_input,
            max_new_tokens=150,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.7, 
            top_p=0.95, 
            top_k=50
        )

        reply_start = model_input.shape[-1] # get input length
        reply_token_ids = output[:, reply_start:]  # get reply_token_ids
        decoded_reply = self.decode_reply(reply_token_ids[0]) # return decoded reply
        
        appended_reply_ids = self.encode_prompt(decoded_reply + "<|end|>\n") # add end symbol to reply before encoding
        self.chat_history_ids = torch.cat([model_input, appended_reply_ids], dim=1) # update chat_history_ids
        
        return decoded_reply


    



