
import torch
from data import image_emb_size, sequence_length, label_to_ignore, tokenizer
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from timeit import default_timer as timer

def timeit():
    torch.cuda.synchronize()
    return timer()
            
class ImgToTextHfLlama2Decoder(torch.nn.Module):
    def __init__(self, image_n_emb=image_emb_size, decoder_n_emb=sequence_length):
        super(ImgToTextHfLlama2Decoder, self).__init__()
        self.decoder = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.decoder_emb_table = self.decoder.model.embed_tokens
        self.img2token_emb_projector = torch.nn.Linear(image_n_emb, decoder_n_emb)        
        lora_config = self.get_lora_config()
        self.decoder_lora = get_peft_model(self.decoder, lora_config)
    
    def get_lora_config(self):
        lora_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        # TODO: Specify all SA and FNN layers here.
        return lora_config

    def forward(self, image_emb, token_ids, target_ids):
        image_emb = self.img2token_emb_projector(image_emb).unsqueeze(1)

        # Get token embeddings manually, to concat image embedding before it.
        token_input_embeds = self.decoder_emb_table(token_ids['input_ids'])
        inputs_embeds = torch.cat([image_emb, token_input_embeds], dim=1)
        
        # Update attention mask with 1s in first element for each sequence of the batch.
        attention_mask = torch.cat([torch.ones((token_ids['input_ids'].shape[0], 1), device='cuda'), token_ids['attention_mask']], dim=1)
        # Add -100 before target IDs - note that HF's LLama2 shifts labels right and ignores first element. Similarly it ignores last logits,
        if target_ids is not None:
            target_ids = torch.cat([label_to_ignore * torch.ones((token_ids['input_ids'].shape[0], 1), device='cuda', dtype=torch.int), target_ids], dim=1)
        
        out =  self.decoder_lora(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=target_ids, return_dict=True)
        loss, logits = out["loss"], out["logits"]
        return loss, logits
    
    def generate(image_emb, model_path, temperature):
        model = torch.load(model_path)
        decoder_lora = model.decoder_lora
        image_emb = model.img2token_emb_projector(image_emb).unsqueeze(1)

        # Get token embeddings manually, to concat image embedding before it.
        # token_input_embeds = self.decoder_emb_table(token_ids['input_ids'])
        # inputs_embeds = torch.cat([image_emb, token_input_embeds], dim=1)
        
        # Update attention mask with 1s in first element for each sequence of the batch.
        # attention_mask = torch.cat([torch.ones((token_ids['input_ids'].shape[0], 1), device='cuda'), token_ids['attention_mask']], dim=1)
        # Add -100 before target IDs - note that HF's LLama2 shifts labels right and ignores first element. Similarly it ignores last logits,
        # if target_ids is not None:
        #     target_ids = torch.cat([label_to_ignore * torch.ones((token_ids['input_ids'].shape[0], 1), device='cuda', dtype=torch.int), target_ids], dim=1)
        should_continue = True
        kv_cache = None
        while should_continue:
            out =  decoder_lora(inputs_embeds=image_emb, use_cache=True, return_dict=True, past_key_values=kv_cache)
            logits, kv_cache = out["loss"], out["past_key_values"]
            last_logits = logits[:, -1, :]
            if temperature > 0.0:
                last_logits = last_logits / temperature
                next_token_id = torch.multinomial(F.softmax(last_logits, dim=-1), num_samples=1)
            else:
                next_token_id = torch.argmax(last_logits, dim=-1)
            if next_token_id != tokenizer.eos_token_id:
                print(tokenizer.decode(next_token_id))
                print(" ")
            else:
                should_continue = False
                

class FakeModel(torch.nn.Module):
    def __init__(self, image_n_emb, decoder_n_emb):
        super(FakeModel, self).__init__()
        self.x = torch.nn.Linear(1,1, bias=False)

    def forward(self, image_emb, idx):
        device = image_emb.device
        x1 = timeit()
        val = torch.matmul(image_emb, idx)
        loss = self.x(val.sum().unsqueeze(0))
        logits = 15.8
        x2 = timeit()
        print("**********************")
        print(f'In MODEL --> total time = {x2-x1}')
        print("**********************")
        return logits, loss
    
