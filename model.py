from transformers import GPT2LMHeadModel
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import clip
from PIL import Image
from copy import deepcopy

features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()#.requires_grad(True)
    return hook

class image_encoder():
    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
    
    def encode(self, images):
        '''
        input: type->list, shape->(batch, image_paths)
        output: type->torch tensor, shape->(batch, images*(patches+1), dimension)
        '''
        output = []
        self.clip_model.visual.transformer.register_forward_hook(get_features('feats'))
        for batch in images:
            batch_output = torch.tensor([], device=self.device, requires_grad=True)
            for image_path in batch:
                image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                self.clip_model.encode_image(image)
                feature = features['feats'].permute(1,0,2)
                feature = feature.float().squeeze(0)
                batch_output = torch.cat([batch_output, feature], dim=0)
            output.append(batch_output)
        
        output = pad_sequence(output, batch_first=True)
        return output

class text_encoder():
    def __init__(self, device):
        self.device = device
        self.wte = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte
        self.ite = nn.Embedding(1, self.wte.weight.shape[-1], device=self.device)
        # train only image token embedding
        self.wte.requires_grad_(False)
        self.ite.requires_grad_(True)

    def encode(self, tokenized_texts):
        '''
        input: type->list shape->(batch, texts, token_ids)
        output: text_embeddings - type->torch.tensor, shape->(batch, texts*words, dimension)
                mask - type->torch.tensor, shape-> (batch, texts*words)
        '''
        output = []
        output_mask = []
        # *image token is always at the first of the sentence
        for idx, batch in enumerate(tokenized_texts):
            text_embeddings = torch.tensor([], device=self.device, requires_grad=True)
            mask = torch.tensor([], device=self.device)
            for text_idx, tokenized_text in enumerate(batch):
                if tokenized_text[0] == self.wte.weight.shape[0]:
                    image_token_embedding = self.ite(torch.tensor([0], device=self.device))
                    text_embeddings = torch.cat([text_embeddings, image_token_embedding], dim=0)
                    tokenized_text = tokenized_text[1:]
                    mask = torch.cat([mask, torch.tensor([text_idx], device=self.device)], dim=0)

                tokenized_text = torch.tensor(tokenized_text)
                history_embedding = self.wte(tokenized_text)
                history_embedding = history_embedding.to(self.device)
                text_embeddings = torch.cat([text_embeddings, history_embedding], dim=0)
    
                text_mask = torch.ones([len(tokenized_text)], device=self.device)*text_idx
                mask = torch.cat([mask, text_mask], dim=0)
                mask = mask.type(torch.int8)
            
            output.append(text_embeddings)
            output_mask.append(mask)
            
        output = pad_sequence(output, batch_first=True)
        output_mask = pad_sequence(output_mask, batch_first=True, padding_value=-1)
        return output, output_mask

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = dim * mult
        self.ffw = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False)
            )

    def forward(self, x):
        # input dimension: [length, dimension]
        # output dimension: [length, dimension]
        return self.ffw(x)
    
class MaskedCrossAttention(nn.Module): 
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, y, mask):
        '''
        input
        x: ([batch, texts*words, dimension]
        y: [batch, images * (num_patches+1), dimension]])
        mask: binary matrix [batch, texts * words, images * (num_patches+1)]
        output: [num_sentence * max_words, dimension])
        '''
        q = self.to_q(x)*self.scale
        k = self.to_k(y)
        v = self.to_v(y)
        q = torch.split(q, self.dim_head, dim=-1)
        k = torch.split(k, self.dim_head, dim=-1)
        v = torch.split(v, self.dim_head, dim=-1)

        attn = torch.tensor([], device=self.device, requires_grad=True)

        for query, key, value in zip(q, k, v):
            attn_val = torch.matmul(query, torch.transpose(key, 1, 2)) + mask * -1e9
            attn_val = attn_val.softmax(dim=-1)
            attn_val = torch.matmul(attn_val, value)
            attn = torch.cat([attn, attn_val], dim=-1)
        output = self.to_out(attn)

        return output

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, ffw_mult):
        super().__init__()
        self.mca = MaskedCrossAttention(dim, dim_head, heads)
        self.mca_gate = nn.Parameter(torch.tensor([0.]))
        self.ffw = FeedForward(dim, ffw_mult)
        self.ffw_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, y, mask): # x: text_embeddings, y: image_embeddings
        x = x + self.mca(x, y, mask) * self.mca_gate.tanh()
        x = x + self.ffw(x) * self.ffw_gate.tanh()
        return x

class GenDial(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device=device
        self.image_patches = 50
        self.text_encoder = text_encoder(self.device)
        self.image_encoder = image_encoder(self.device)
        self.cross_attn = GatedCrossAttentionBlock(dim = 768, dim_head = 64, heads = 8, ffw_mult=4).to(self.device)
        self.lm = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        
        for param in self.lm.parameters():
            param.requires_grad=False
        
    def create_mask(self, mask_index, image_embedding_length):
        '''
        input: mask_index, image_embedding_length
        mask_index -> torch.tensor, [batch, texts*words]
        image_embedding_length -> scalar
        num_image_patch -> scalar
        output: mask - type-> torch.tensor, shape->[batch, texts*words, images*(patches+1)]
        '''
        num_batch = mask_index.shape[0]
        text_embedding_length = mask_index.shape[1]

        mask = torch.ones([num_batch, text_embedding_length, image_embedding_length], device=self.device)
        for batch in range(num_batch):
            for idx, num_preceding_image in enumerate(mask_index[batch]):
                if num_preceding_image != -1:
                    mask[batch, idx, :(num_preceding_image+1) * self.image_patches] = 0
        return mask

    def input_to_embed(self, data):
        '''
        input:  type-> image_tokens / text_tokens / response_tokens, shape
                shape-> image_tokens - [batch, images]
        '''
        image_embeddings = self.image_encoder.encode(data['image'])
        text_embeddings, mask_index = self.text_encoder.encode(data['text'])
        response_tokens = data['response']
        mask = self.create_mask(mask_index, image_embeddings.shape[1])

        return {'image':image_embeddings, 'text': text_embeddings, 'response': response_tokens, 'mask': mask}

    def forward(self, data):
        '''
        input: type-> dictionary of tokens {image, text, response}
               shape->  image - list of batch of image paths
                        text - list of batch of text tokens
                        response - list of batch of response tokens
        output: list of prediction per text
                shape -> [batch, reponse_length]
        '''
        prediction = []
        # predict
        input = self.input_to_embed(data)

        image_text_feature = self.cross_attn(input['text'], input['image'], input['mask'])
        
        output = self.lm(inputs_embeds = image_text_feature).logits # shape [batch, max_length, vocab_length]

        return output