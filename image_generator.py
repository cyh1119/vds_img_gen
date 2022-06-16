import jax
import jax.numpy as jnp
from functools import partial
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate
import random
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key, shard
import numpy as np
from PIL import Image
from tqdm import trange
from transformers import CLIPProcessor, FlaxCLIPModel

class dall_e:
    def __init__(self):
        DALLE_MODEL = "dalle-mini/dalle-mini/mega-1:latest"
        DALLE_COMMIT_ID = None

        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

        CLIP_REPO = "openai/clip-vit-base-patch32"
        CLIP_COMMIT_ID = None

        print('loading dalle-mega...')
        # Load dalle-mini
        self.model, self.params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float32, _do_init=False
        )

        print('loading VQGAN...')
        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )
    
        # Load Tokenizer
        print('loading tokenizer...')
        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
        
        # Load CLIP
        print('loading CLIP...')
        self.clip, self.clip_params = FlaxCLIPModel.from_pretrained(
            CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)

        self.params = replicate(self.params)
        self.vqgan_params = replicate(self.vqgan_params)
        self.clip_params = replicate(self.clip_params)

        # create a random key
        seed = random.randint(0, 2**32 - 1)
        self.key = jax.random.PRNGKey(seed)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4, 5, 6, 7))
    def p_generate(self, tokenized_text, key, params, top_k, top_p, temperature, condition_scale):
        return self.model.generate(
            **tokenized_text,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )


    # decode image
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0))
    def p_decode(self, indices, params):
        return self.vqgan.decode_code(indices, params=params)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0))
    def p_clip(self, inputs, params):
        logits = self.clip(params=params, **inputs).logits_per_image
        return logits

    def generate_images(self, text, n_predictions = 1, gen_top_k = None, gen_top_p = None, temperature = None, cond_scale = 10.0):
        images = []
        
        tokenized_text = self.processor([text])
        tokenized_text = replicate(tokenized_text)

        # generate images
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # get a new key
            self.key, subkey = jax.random.split(self.key)
            # generate images
            encoded_images = self.p_generate(
                tokenized_text,
                shard_prng_key(subkey),
                self.params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = self.p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
        return images

    def generate_best_image(self, text, n_predictions = 8, gen_top_k = None, gen_top_p = None, temperature = None, cond_scale = 10.0):
        images = self.generate_images(text, n_predictions, gen_top_k, gen_top_p, temperature, cond_scale)
        # get clip scores
        clip_inputs = self.clip_processor(
            text=[text] * jax.device_count(),
            images=images,
            return_tensors="np",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).data
        logits = self.p_clip(shard(clip_inputs), self.clip_params)
        logits = logits.squeeze().flatten()

        return images[np.argmax(logits.argsort())]
        
