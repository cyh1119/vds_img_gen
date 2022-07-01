from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import clip
from PIL import Image
from torch import nn

features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

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
        x: ([num_sentence * max_words, dimension]
        y: [num_sentence, * (num_patches+1), dimension]])
        mask: binary matrix about if each word can attend ([num_sentence * max_words, num_sentence * (num_patches+1)])
        output: [num_sentence * max_words, dimension])
        '''
        q = self.to_q(x)*self.scale
        k = self.to_k(y)
        v = self.to_v(y)
        q = torch.split(q, self.dim_head, dim=1)
        k = torch.split(k, self.dim_head, dim=1)
        v = torch.split(v, self.dim_head, dim=1)

        attn = torch.tensor([], device=self.device, requires_grad=True)

        for query, key, value in zip(q, k, v):
            attn_val = torch.matmul(query, key.T) + mask * -1e9
            attn_val = attn_val.softmax(dim=-1)
            attn_val = torch.matmul(attn_val, value)
            attn = torch.cat([attn, attn_val], dim=1)

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
        print(self.ffw_gate.grad)
        return x

class encoder():
    def __init__(self, wte, device):
        self.device=device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token="<|image|>")
        for clip_params in self.clip_model.parameters():
            clip_params.requires_grad = False
        self.wte = wte
        self.image_token_embedding = nn.Embedding(1, wte.weight.shape[-1], device=self.device)
        # train only image token embedding
        self.wte.requires_grad_(False)
        self.image_token_embedding.requires_grad_(True)

    def encoding_image(self, image_path_list):  # output: [num_images, num_patches+1, dimension]
        output = torch.Tensor([]).to(self.device)
        self.clip_model.visual.transformer.register_forward_hook(get_features('feats'))
        for image_path in image_path_list:
            image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

            self.clip_model.encode_image(image)

            feature = features['feats'].permute(1,0,2)
            feature = feature.float()
            output = torch.cat([output, feature], dim=-2)
        return output.squeeze(0)

    def tokenize_text(self, texts): # output: list of text tokens
        tokens = []
        for text in texts:
            token = self.tokenizer(text, return_tensors='pt')
            input_ids = token['input_ids'].to(self.device)
            tokens.append(input_ids)
        return tokens

    def encoding_text(self, text_tokens, ): # input: list of text tokens, output: tensor of text embeddings
        text_embeddings = torch.tensor([],device=self.device, requires_grad=True)

        # *image token is always at the first of the sentence
        for text_token in text_tokens:
            if text_token[0][0] == self.wte.weight.shape[0]:
                image_token_embed = self.image_token_embedding(torch.tensor([0], device=self.device))
                text_embeddings = torch.cat([image_token_embed, text_embeddings], dim=0)
                text_token = text_token[0][1:]
                text_token = torch.tensor(text_token, device=self.device)
            text_embeddings = torch.cat([text_embeddings, self.wte(text_token)], dim=-2)

        return text_embeddings

class GenDial(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device=device
        self.lm = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.wte = self.lm.transformer.wte
        self.encoder = encoder(self.wte, self.device)
        self.cross_attn = GatedCrossAttentionBlock(dim = 768, dim_head = 64, heads = 8, ffw_mult=4).to(self.device)

        for param in self.lm.parameters():
            param.requires_grad=False
        
    def create_mask_per_text(self, num_text, num_words, num_preceding_image, num_embedds):
        mask = torch.ones((num_words, num_text * num_embedds))
        mask[:, :num_preceding_image * num_embedds] = 0
        return mask

    def forward(self, data):
        image_path = data['images']
        texts = data['history']
        response = data['response']

        # remove None
        image_path = [image for image in image_path if not image == None]
        texts = [text for text in texts if not text == None]

        # tokenize text and response
        text_tokens = self.encoder.tokenize_text(texts)
        response_tokens = self.encoder.tokenize_text(response)[0].squeeze()
        
        # encode image and text
        image_embeddings = self.encoder.encoding_image(image_path)
        text_embeddings = self.encoder.encoding_text(text_tokens)

        # create mask matrix
        mask = torch.Tensor([]).to(self.device)
        num_embedds = int(image_embeddings.shape[-2] / len(texts))
        for idx, text_token in enumerate(text_tokens):
            mask_per_text = self.create_mask_per_text(len(texts), text_token.shape[-1], idx+1 ,num_embedds).to(self.device)
            mask = torch.cat([mask, mask_per_text], dim=0)

        prediction = torch.tensor([], device=self.device, requires_grad=True)

        # predict next words
        for iteration in range(response_tokens.shape[-1]):
            image_text_feature = self.cross_attn(text_embeddings, image_embeddings, mask)
            
            output = self.lm(inputs_embeds = image_text_feature).logits
            
            prob = torch.softmax(output[-1], dim=-1).unsqueeze(0)
            prediction = torch.cat([prediction, prob], dim=0)
            new_token = [torch.tensor(response_tokens[iteration], device=self.device).unsqueeze(0).unsqueeze(0)]
            new_token_embedding = self.encoder.encoding_text(new_token).squeeze(0)
            text_embeddings = torch.cat([text_embeddings, new_token_embedding], dim=0)

            # modify mask matrix
            mask = torch.cat([mask, torch.zeros((1, len(texts) * num_embedds), device=self.device)], dim=0)

        return prediction, response_tokens