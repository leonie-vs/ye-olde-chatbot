from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    
    def __init__(self):

        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.system_prompt_ids = self.tokenizer("<|system|>\nYou are a helpful assistant.\n<|end|>\n", return_tensors="pt")["input_ids"].to(self.device)
        self.chat_history_ids = None

    def reset_history(self):
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt_text: str) -> str:
        
        prompt = f"<|user|>\n{prompt_text}\n<|end|>\n<|assistant|>\n"
        input_ids = self.encode_prompt(prompt) # encode prompt to get input ids
        
        if self.chat_history_ids is None: # check if this is the first prompt
            model_input = torch.cat([self.system_prompt_ids, input_ids], dim=1) # pass prompt ids combined with system_prompt ids
        else: # if not first, pass prompt combined with chat_history_ids
            model_input = torch.cat([self.chat_history_ids, input_ids], dim=1)

        attention_mask = torch.ones_like(model_input).to(self.device)

        output = self.model.generate( # generate reply 
            model_input,
            attention_mask=attention_mask,
            max_new_tokens=150,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"),
            do_sample=True, 
            temperature=0.7, 
            top_p=0.95, 
            top_k=50
        )

        reply_start = model_input.shape[-1] # get input length
        reply_token_ids = output[:, reply_start:]  # get reply_token_ids

        reply_text = self.decode_reply(reply_token_ids[0]) # decode reply

        for token in ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]:
            reply_text = reply_text.split(token)[0] # tidy up reply text
        
        reply_text = reply_text.strip()
        
        # update chat history ids
        end_ids = self.encode_prompt("<|end|>\n")
        if self.chat_history_ids is None:
            self.chat_history_ids = torch.cat([
                self.system_prompt_ids, input_ids, reply_token_ids, end_ids
            ], dim=1)
        else:
            self.chat_history_ids = torch.cat([
                self.chat_history_ids, input_ids, reply_token_ids, end_ids
            ], dim=1)
        
        return reply_text # return decoded reply


    



