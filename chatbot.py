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
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        
        prompt = prompt + "\n" # append newline character
        encoded_input = self.encode_prompt(prompt) # encode prompt
        input_ids = encoded_input["input_ids"] # get input ids

        if self.chat_history_ids is None: # check if this the first prompt
            system_prompt_ids = self.encode_prompt(self.system_prompt)["input_ids"] # if yes, get system prompt tokens
            model_input = torch.cat([system_prompt_ids, input_ids], dim=1) # pass prompt combined with the system_prompt
        else:
            model_input = torch.cat([self.chat_history_ids, input_ids], dim=1) # if not first, pass prompt combined with chat_history_ids

        output = self.model.generate( # generate reply 
            model_input,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.9, 
            top_p=0.8, 
            top_k=50
        )

        self.chat_history_ids = output # update chat_history_ids
    
        reply_start = model_input.shape[-1] 
        new_tokens = output[:, reply_start:] 
        reply_token_ids = new_tokens # get reply_token_ids
        
        return self.decode_reply(reply_token_ids[0]) # return decoded reply 
        


    



