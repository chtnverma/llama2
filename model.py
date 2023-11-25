
import torch
from gpt import GPTConfig, GPT
from data import padding_idx, eos_token_idx, image_emb_size, sequence_length
from transformers import GPT2LMHeadModel, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

from timeit import default_timer as timer

def timeit(device):
    if True: #device == torch.device("mps"):
        torch.cuda.synchronize()
    return timer()


# class ImgToTextDecoder(torch.nn.Module):
#     def __init__(self, image_n_emb, decoder_n_emb):
#         super(ImgToTextDecoder, self).__init__()
#         gptconf = GPTConfig(n_embd=decoder_n_emb)
#         self.decoder = GPT(gptconf)
#         self.img2token_emb_projector = torch.nn.Linear(image_n_emb, decoder_n_emb)
#         # TODO(Chetan): look at initialization

#     def forward(self, image_emb, idx):
#         device = image_emb.device
#         x1 = timeit(device)
#         image_emb = self.img2token_emb_projector(image_emb)
#         # print(f"### Shape of idx = {idx.shape}")
#         x2 = timeit(device)
#         targets = idx
#         inputs = idx[:, :-1]
#         x3 = timeit(device)
#         inputs[inputs == padding_idx] = enc.eot_token
#         x4 = timeit(device)
#         out =  self.decoder(idx=inputs, targets=targets, initial_emb=image_emb)
#         x5 = timeit(device)

#         print("**********************")
#         print(f'In MODEL --> total time = {x5-x1}')
#         print(f'img2token_emb_projector: {x2-x1}')
#         print(f'create index by remove last: {x3-x2}')
#         print(f'Add padding (eot) token: {x4-x3}')
#         print(f'Decoder (gpt) time: {x5-x4}')
#         print("**********************")

#         return out


# class ImgToTextHfDecoder(torch.nn.Module):
#     def __init__(self, image_n_emb=image_emb_size, decoder_n_emb=768):
#         super(ImgToTextHfDecoder, self).__init__()
#         # gptconf = GPTConfig(n_embd=decoder_n_emb)
#         self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
#         self.decoder_emb_table = self.decoder._modules['transformer'].wte
#         self.img2token_emb_projector = torch.nn.Linear(image_n_emb, decoder_n_emb)
#         # TODO(Chetan): look at initialization

#     def forward(self, image_emb, token_ids):
#         image_emb = self.img2token_emb_projector(image_emb).unsqueeze(1)
#         # targets = idFx
#         # inputs = idx[:, :-1]
#         # inputs = idx
#         token_ids = torch.where(token_ids != padding_idx, token_ids, eos_token_id * torch.ones_like(token_ids))
#         token_input_embeds = self.decoder_emb_table(token_ids)
#         # print("Shape of image_emb = ", image_emb.shape)
#         # print("Shape of token_input_embeds = ", token_input_embeds.shape)
#         # inputs[inputs == padding_idx] = enc.eot_token # TODO: get mask index from this.
#         concat_input_embs = torch.cat([image_emb, token_input_embeds], dim=1)
#         out =  self.decoder(inputs_embeds=concat_input_embs)
#         # print("----- [start] printing output of gpt model -----")
#         # print(f'Length of output of gpt = {len(out)}')
#         # # print(out[0])
#         # print(out[0].shape)
#         # print(len(out[1]))
#         # print("----- [end] printing output of gpt model -----")
#         logits, loss = out[0], None
#         return (logits, loss)

#     def train_val_step(self, logits, loss, token_ids, train=True):
#         if train:
#             self.train()
#         else:
#             self.eval()
        
#         if loss is None:
#             # Control comes here for soft-prompted GPT output:
#             # print("PRINTING THINGS INSIDE TRAIN_STEP() ---")
#             # print(logits.shape)
#             # print(token_ids.shape)
#             # print("----------------------------------------")
#             logits = logits[:, :-1, :].transpose(1, 2)
#             ce_loss = torch.nn.CrossEntropyLoss()
#             loss = ce_loss(input=logits, target=token_ids)
            
#         if train:
#             loss.backward()

#         return loss
            
class ImgToTextHfLlama2Decoder(torch.nn.Module):
    def __init__(self, image_n_emb=image_emb_size, decoder_n_emb=sequence_length, padding_idx=padding_idx, eos_token_idx=eos_token_idx):
        # TODO: earlier padding_idx was -100. Handle that in label calculation.
        super(ImgToTextHfLlama2Decoder, self).__init__()
        self.decoder = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.decoder.resize_token_embeddings(self.decoder.config.vocab_size + 1) # for newly added padding token
        self.decoder_emb_table = self.decoder.model.embed_tokens
        self.img2token_emb_projector = torch.nn.Linear(image_n_emb, decoder_n_emb)
        self.padding_idx = padding_idx
        # self.eos_token_idx = eos_token_idx
        
        lora_config = self.get_lora_config()
        self.decoder_lora = get_peft_model(self.decoder, lora_config)
        
        
    
    def get_lora_config(self):
        lora_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        # TODO: which layers are being replaced? And how to specify all SA and FNN layers?
        return lora_config

    def forward(self, image_emb, token_ids):
        image_emb = self.img2token_emb_projector(image_emb).unsqueeze(1)

        # NOT NEEDED? -- Padding token isn't in embedding layer, so replace with EOS. This will be masked out via attention mask -- TO VERIFY!
        # token_ids['input_ids'] = torch.where(token_ids['input_ids'] != self.padding_idx, token_ids['input_ids'], self.eos_token_idx * torch.ones_like(token_ids['input_ids']))
        
        # Get token embeddings manually, to concat image embedding before it.
        token_input_embeds = self.decoder_emb_table(token_ids['input_ids'])
        # print("In forward() --  input_ids = ", token_ids['input_ids'])
        # print("In forward() --  input_ids.shape = ", token_ids['input_ids'].shape)
        

        concat_input_embs = torch.cat([image_emb, token_input_embeds], dim=1)
        
        # Update attention mask with 1s in first column.
        attention_mask = torch.cat([torch.ones((token_ids['input_ids'].shape[0], 1), device='cuda'), token_ids['attention_mask']], dim=1)
        # print("Shape of attention_mask (post-surgery) = ", attention_mask.shape)
        out =  self.decoder_lora(inputs_embeds=concat_input_embs, attention_mask=attention_mask)
        logits, loss = out[0], None
        return (logits, loss)

    def train_val_step(self, logits, loss, token_ids, train=True):
        if train:
            self.decoder_lora.train()
        else:
            self.decoder_lora.eval()
    
        # print("-----------------")
        # print("In train_val_step() --  input_ids = ", token_ids['input_ids'])
        # print("In train_val_step() --  input_ids.shape = ", token_ids['input_ids'].shape)
        
        # input() 
        
        # print("CHETAN DEGUB. BEFORE logits = ", logits.shape)
        logits = logits[:, :-1, :].transpose(1, 2)
        # print("CHETAN DEGUB. logits = ", logits.shape, logits)
        target = torch.where(token_ids['input_ids'] != self.padding_idx, token_ids['input_ids'], -100 * torch.ones_like(token_ids['input_ids']))
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(input=logits, target=target)
            
        if train:
            loss.backward()

        return loss
            


class FakeModel(torch.nn.Module):
    def __init__(self, image_n_emb, decoder_n_emb):
        super(FakeModel, self).__init__()
        self.x = torch.nn.Linear(1,1, bias=False)

    def forward(self, image_emb, idx):
        device = image_emb.device
        x1 = timeit(device)
        val = torch.matmul(image_emb, idx)
        loss = self.x(val.sum().unsqueeze(0))
        logits = 15.8
        x2 = timeit(device)
        print("**********************")
        print(f'In MODEL --> total time = {x2-x1}')
        print("**********************")
        return logits, loss
    
