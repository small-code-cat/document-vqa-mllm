import fitz
import os

from config import RAG_CONFIG
import uuid
from box import Box
from doclayout_yolo import YOLOv10
import base64
from PIL import Image
from tqdm import tqdm
import hashlib
from collections import defaultdict
from pathlib import Path
from transformers import BatchEncoding
from contextlib import nullcontext
from transformers import AutoTokenizer as Tokenizer_class
from transformers import AutoModel as Model_class
from transformers import AutoModelForCausalLM as ModelForCausalLM_class
from openai import OpenAI
import gc
import streamlit as st
import pickle
import torch
import io
import numpy as np

def get_stable_pdf_id(file):
    file.seek(0)
    content = file.read()
    file.seek(0)  # 重置读取位置
    return hashlib.md5(content).hexdigest()

def pdf_to_images_mupdf(uploaded_file, output_folder=None, zoom_x=2.0, zoom_y=2.0, fmt='png', is_save=True):
    img_list = []
    # 用 fitz 读取上传的 PDF
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # 加载页面

            # 设置缩放比例（2.0 相当于放大 200%，用于提高清晰度）
            mat = fitz.Matrix(zoom_x, zoom_y)

            pix = page.get_pixmap(matrix=mat)
            if is_save:
                image_path = os.path.join(output_folder, f'{page_num + 1}.{fmt}')
                pix.save(image_path)
            else:
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                img_list.append(img)
    return None if is_save else img_list

def get_tokenizer(retriever):
    if retriever == 'VisRAG-Retriever(MiniCPM-V2.0)':
        from model.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls

    tokenizer = tokenizer_cls.from_pretrained(
        RAG_CONFIG['retrievers'][retriever]['path'],
        cache_dir=None,
        use_fast=False
    )

    return tokenizer

@st.cache_resource(show_spinner=False)
def get_model(retriever):
    if retriever == 'VisRAG-Retriever(MiniCPM-V2.0)':
        from model.modeling import DRModelForInference
        model = DRModelForInference.build(
            model_args=Box({
                'model_name_or_path': RAG_CONFIG.retrievers[retriever].path,
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

    model.to(RAG_CONFIG.device)
    model.eval()
    return model

def load_images_in_batches(target_dir, batch_size=16, extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    """
    从文件夹中按批次加载图片
    返回每一批为一个列表，包含：(文件名, PIL Image) 元组
    """
    all_files = sorted([
        f for f in os.listdir(target_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    total = len(all_files)
    batches = []

    for i in range(0, total, batch_size):
        batch_files = all_files[i:i + batch_size]
        batch = defaultdict(list)
        for filename in batch_files:
            file_path = os.path.join(target_dir, filename)
            try:
                image = Image.open(file_path).convert("RGB")
                batch['id'].append(filename)
                batch['text'].append('')
                batch['image'].append(image)
            except Exception as e:
                print(f"跳过 {filename}，读取失败: {e}")
        batches.append(batch)

    return batches

def to_device(data, device):
    """
    Recursively move tensors in a nested list, tuple, or dictionary to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    elif isinstance(data, BatchEncoding):
        return data.to(device)
    else:
        return data

def encode_corpus_by_visrag(image_dir, output_dir, tokenizer, model, granularity):
    encoded = []
    lookup_indices = []
    data_args = Box({
        'p_max_len': 2048,

    })
    model_additional_args = {  # for VisRAG_Ret only
        "tokenizer": tokenizer,
        "max_inp_length": data_args.p_max_len
    }
    batches = load_images_in_batches(image_dir)
    for batch in tqdm(batches):
        batch_ids = batch['id']
        lookup_indices.extend(batch_ids)
        with nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = to_device(v, RAG_CONFIG.device)  # a BatchEncoding object, can contain strings.

                model_output = model(passage=batch, **model_additional_args)
                encoded_ = model_output.p_reps.cpu().detach().numpy()
                encoded.append(encoded_)
    encoded = np.concatenate(encoded)
    index_path = os.path.join(output_dir, "{}-{}-embeddings".format(Path(image_dir).parent.name, granularity))
    with open(index_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f, protocol=4)
    return index_path

def encode_query(query, tokenizer, model):
    encoded = []
    data_args = Box({
        'p_max_len': 2048,

    })
    model_additional_args = {  # for VisRAG_Ret only
        "tokenizer": tokenizer,
        "max_inp_length": data_args.p_max_len
    }
    batch = defaultdict(list)
    batch['id'].append('')
    batch['text'].append(RAG_CONFIG.templates.visrag_query_template.format(query=query))
    batch['image'].append(None)
    model_output = model(query=batch, **model_additional_args)
    encoded_ = model_output.q_reps.cpu().detach().numpy()
    encoded.append(encoded_)
    encoded = np.concatenate(encoded)
    return encoded


def _retrieve_one_shard(
        corpus_shard_path: str,
        encoded_queries_tensor: torch.Tensor,
        topk: int,
        device: str,
):
    with open(corpus_shard_path, "rb") as f:
        data = pickle.load(f)
    encoded_corpus = data[0]
    corpus_lookup_indices = data[1]

    encoded_corpus_tensor = torch.tensor(encoded_corpus, device=device)

    # compute the inner product of the query and corpus
    scores = torch.matmul(encoded_queries_tensor, encoded_corpus_tensor.T)

    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)

    del encoded_corpus, encoded_corpus_tensor, scores
    gc.collect()
    return topk_scores.clone(), topk_indices.clone(), corpus_lookup_indices

def get_corpus(query_embedding, index_path):
    query_embedding = torch.tensor(query_embedding, device=RAG_CONFIG.device)
    topk_scores, topk_indices, corpus_lookup_indices = _retrieve_one_shard(
        corpus_shard_path=index_path,
        encoded_queries_tensor=query_embedding,
        topk=RAG_CONFIG.topk,
        device=RAG_CONFIG.device
    )
    result = [corpus_lookup_indices[idx.item()] for idx, _ in zip(topk_indices[0], topk_scores[0])]
    return result

@st.cache_resource(show_spinner=False)
def get_minicpmv2_6(generator_name):
    model = Model_class.from_pretrained(RAG_CONFIG.generators[generator_name].path, trust_remote_code=True,
                                        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = Tokenizer_class.from_pretrained(RAG_CONFIG.generators[generator_name].path, trust_remote_code=True)
    return model, tokenizer

def generate_response(query, topk_image_paths, model, tokenizer):
    image_list = [Image.open(i) for i in topk_image_paths]
    input = [
        {'role': 'user', 'content': RAG_CONFIG.templates.visrag_response_template.format(query=query)}]
    input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
    responds = model.chat(
        image=None,
        msgs=input,
        tokenizer=tokenizer,
        sampling=False,
        max_new_tokens=20
    )
    return responds

def empty_memory(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def image_to_base64(image_input, format="png") -> str:
    """
    将多种类型的图像输入转换为 base64 编码字符串。

    支持输入：
        - 文件路径（str）
        - PIL.Image.Image
        - bytes（图像二进制）
        - io.BytesIO
        - NumPy 数组（自动转 PIL）
    """
    if isinstance(image_input, str):  # 文件路径
        with open(image_input, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_input, bytes):
        image_bytes = image_input
    elif isinstance(image_input, io.BytesIO):
        image_bytes = image_input.getvalue()
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.convert("RGB").save(buffer, format=format)
        image_bytes = buffer.getvalue()
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=format)
        image_bytes = buffer.getvalue()
    else:
        raise TypeError("Unsupported image_input type.")

    return base64.b64encode(image_bytes).decode("utf-8")


def build_message_content(prompt: str, image_base64_list: list):
    content = [{"type": "text", "text": prompt}]

    for base64_img in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    return content

def get_client():
    client = OpenAI(
        # 将这里换成你在便携AI聚合API后台生成的令牌
        api_key=RAG_CONFIG.api_key,
        # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
        base_url=RAG_CONFIG.base_url,
    )
    return client

def generate_gemini_response(query: str, image_list, client, model_name) -> str:
    image_base64_list = [image_to_base64(i) for i in image_list]
    message_content = build_message_content(RAG_CONFIG.templates.gemini_resposne_template.format(query=query), image_base64_list)

    messages = [
        {
            "role": "user",
            "content": message_content
        }
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return completion.choices[0].message.content.strip()

def get_layout_model():
    model = YOLOv10(RAG_CONFIG.layout_model.path)
    return model

def standard_detection_result(detection_result, abandon_labels={'abandon'}):
    orig_img, boxes, names_dict = detection_result.orig_img, detection_result.boxes, detection_result.names

    if isinstance(orig_img, np.ndarray):
        image = Image.fromarray(orig_img[..., ::-1].astype(np.uint8))
    else:
        raise ValueError("Unsupported image format")

    cls = boxes.cls.cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()

    # 存储所有框信息，保留顺序
    all_regions = []
    for label_id, box in zip(cls, xyxy):
        label = names_dict[int(label_id)]
        if label in abandon_labels:
            continue
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))
        all_regions.append(cropped)

    return all_regions

def parse_pdf_by_layout(uploaded_file, output_folder, detection_conf=0.3, fmt='png'):
    model = get_layout_model()
    img_list = pdf_to_images_mupdf(uploaded_file, is_save=False)
    for page_num, img in enumerate(img_list):
        det_res = model.predict(
            img,  # Image to predict
            imgsz=1024,  # Prediction image size
            conf=detection_conf,  # Confidence threshold
            device=RAG_CONFIG.device  # Device to use (e.g., 'cuda:0' or 'cpu')
        )
        all_regions = standard_detection_result(det_res[0])
        for region in all_regions:
            image_path = os.path.join(output_folder, f'{page_num + 1}-{str(uuid.uuid4())}.{fmt}')
            region.save(image_path)