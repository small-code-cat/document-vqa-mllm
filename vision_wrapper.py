import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import io
from box import Box
from collections import defaultdict
from config import RAG_CONFIG
import json

from transformers import AutoModelForCausalLM, AutoProcessor


class DSE(nn.Module):
    def __init__(self, model_name="checkpoint/dse-phi3-v1", lora_adapter=None, bs=4, flash_attn=True):
        super().__init__() # "checkpoint/dse-phi3-docmatix-v2" "checkpoint/dse-phi3-v1.0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def set_attention_implementation(model_name, flash_attn):
            config_path = f"{model_name}/config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            config["_attn_implementation"] = "flash_attention_2" if flash_attn else None
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        set_attention_implementation(model_name, flash_attn)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        if lora_adapter:
            self.model = self.model.load_adapter(lora_adapter)
        self.model = self.model.to(self.device)  # First move to primary GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.bs = bs
        self.bs_query = 64

    def embed_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        dataloader = DataLoader(
            queries, batch_size=self.bs_query, shuffle=False,
            collate_fn=lambda xs: self.process_queries(xs)
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[DSE] Embedding queries"):
                reps = self.encode(batch)
                embeddings.extend(reps.cpu().float().numpy())
        return embeddings

    def embed_quotes(self, images, hybrid=False):
        if not hybrid:
            if isinstance(images, (Image.Image, bytes, bytearray)):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs, shuffle=False,
                collate_fn=lambda xs: self.process_images(xs)
            )
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[DSE] Embedding quotes in images"):
                    reps = self.encode(batch)
                    embeddings.extend(reps.cpu().float().numpy())
            return embeddings
        else: # input quotes in text format
            if isinstance(images, str):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs_query, shuffle=False,
                collate_fn=lambda xs: self.process_image_texts(xs)
            )
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[DSE] Embedding quotes in texts"):
                    reps = self.encode(batch)
                    embeddings.extend(reps.cpu().float().numpy())
            return embeddings
        
    def encode(self, batch):
        outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()}, return_dict=True, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        reps = self._pool(hs, batch['attention_mask'].to(self.device))
        return reps

    def _pool(self, last_hidden_state, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        return reps

    def process_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        prompts = [f"query: {q}</s>" for q in queries]
        batch = self.processor(
            prompts,
            images=None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return batch

    def process_image_texts(self, passages):
        if isinstance(passages, str):
            passages = [passages]
        prompts = [f"passage: {p}</s>" for p in passages]
        batch = self.processor(
            prompts,
            images=None,
            return_tensors="pt",
            padding="longest",
            max_length=600,
            truncation=True,
        )
        return batch
    
    def process_images(self, images):
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img.resize((1344, 1344))
            elif isinstance(img, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(img))
                pil_img = pil_img.resize((1344, 1344))
            else:
                raise ValueError("Each image must be a PIL.Image.Image or bytes.")
            pil_images.append(pil_img.convert("RGB"))
        prompts = [f"<|image_{i+1}|>\nWhat is shown in this image?</s>" for i in range(len(pil_images))]
        batch = self.processor(
            prompts,
            images=pil_images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        if batch['input_ids'].dim() == 3: # Squeeze batch dims if needed
            batch['input_ids'] = batch['input_ids'].squeeze(0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(0)
            if 'image_sizes' in batch:
                batch['image_sizes'] = batch['image_sizes'].squeeze(0)
        return batch

    def score(self, query_embs, image_embs):
        query_emb = np.asarray(query_embs)
        quote_emb = np.asarray(image_embs)
        scores = (query_emb @ quote_emb.T).tolist()
        return scores

class VisRAGRetriever:
    def __init__(self, use_gpu=True):
        self.bs = RAG_CONFIG.retrievers['VisRAG-Retriever(MiniCPM-V2.0)'].bs
        self.bs_query = RAG_CONFIG.retrievers['VisRAG-Retriever(MiniCPM-V2.0)'].bs_query
        self.base_ckpt = RAG_CONFIG.retrievers['VisRAG-Retriever(MiniCPM-V2.0)'].path
        device = RAG_CONFIG.retrievers['VisRAG-Retriever(MiniCPM-V2.0)'].device if (torch.cuda.is_available() and use_gpu) else "cpu"
        self.query_prompt = RAG_CONFIG.retrievers['VisRAG-Retriever(MiniCPM-V2.0)'].query_template
        self.tokenizer = self.get_tokenizer()
        self.model = self.__get_model__()
        self.model = self.model.to(device)
        self.model.eval()

    def __get_model__(self):
        from model.modeling import DRModelForInference
        model = DRModelForInference.build(
            model_args=Box({
                'model_name_or_path': self.base_ckpt,
                'attention': 'causal',
                'dtype': 'float16',
                'attn_implementation': 'sdpa',
                'lora': False,
                'add_linear_head': False,
                'feature': 'last_hidden_state',
                'pooling': 'wmean',
                'normalize': True,
            }),
            cache_dir=None,
        )
        return model

    def get_tokenizer(self):
        from model.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls

        tokenizer = tokenizer_cls.from_pretrained(
            self.base_ckpt,
            cache_dir=None,
            use_fast=False
        )

        return tokenizer

    def embed_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        data_args = Box({
            'p_max_len': 2048,

        })
        model_additional_args = {  # for VisRAG_Ret only
            "tokenizer": self.tokenizer,
            "max_inp_length": data_args.p_max_len
        }

        # 构造 DataLoader
        dataloader = DataLoader(
            queries,
            batch_size=self.bs_query,
            shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[VisRAG] Embedding queries"):
                input_samples = defaultdict(list)
                input_samples['id'] = [''] * len(batch)
                input_samples['text'] = [self.query_prompt.format(query=i) for i in batch]
                input_samples['image'] = [None] * len(batch)
                model_output = self.model(query=input_samples, **model_additional_args)
                embeddings.extend(model_output.q_reps.cpu().detach().numpy())

        return embeddings

    def embed_quotes(self, images, hybrid=False):
        data_args = Box({
            'p_max_len': 2048,

        })
        model_additional_args = {  # for VisRAG_Ret only
            "tokenizer": self.tokenizer,
            "max_inp_length": data_args.p_max_len
        }
        if not hybrid:
            embeddings = []
            dataloader = DataLoader(images, batch_size=self.bs, shuffle=False)
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[VisRAG] Embedding quotes in images"):
                    input_samples = defaultdict(list)
                    input_samples['id'] = [''] * len(batch)
                    input_samples['text'] = [''] * len(batch)
                    input_samples['image'] = [Image.open(io.BytesIO(i)) for i in batch]
                    model_output = self.model(passage=input_samples, **model_additional_args)
                    embeddings.extend(model_output.p_reps.cpu().detach().numpy())
            return embeddings
        else: # input quotes in text format
            if isinstance(images, str):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs_query, shuffle=False,
                collate_fn=lambda xs: self.process_image_texts(xs)
            )
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[VisRAG] Embedding quotes in texts"):
                    reps = self.encode(batch)
                    embeddings.extend(reps.cpu().float().numpy())
            return embeddings
