import os.path

from config import RAG_CONFIG
import streamlit as st
from typing import List
from pprint import pprint

import uuid
from utils import *

# -------------------------------
# 模拟后端逻辑函数（你可以替换为真实逻辑）
# -------------------------------
def document_parser(uploaded_file, granularity, target_dir_root='./images'):
    pdf_name = get_stable_pdf_id(uploaded_file)
    if granularity == "layout-level":
        target_dir = os.path.join(target_dir_root, pdf_name, 'layout')
        os.makedirs(target_dir, exist_ok=True)
        parse_pdf_by_layout(uploaded_file, target_dir)
    else:
        target_dir = os.path.join(target_dir_root, pdf_name, 'page')
        os.makedirs(target_dir, exist_ok=True)
        pdf_to_images_mupdf(uploaded_file, target_dir)
    return target_dir

def build_index(image_dir, retriever_name, granularity, index_dir='./indexes'):
    output_dir = os.path.join(index_dir)
    os.makedirs(output_dir, exist_ok=True)
    retriever = get_retriever(retriever_name)
    index_path = encode_corpus(image_dir, output_dir, retriever, granularity)
    return index_path

def retrieve(query: str, retriever_name: str, index_path):
    retriever = get_retriever(retriever_name)
    query_embedding = encode_query(query, retriever)
    result = get_corpus(query_embedding, index_path)
    return result

def generate_answer(topk_image_paths: List[str], query: str, generator_name: str) -> str:
    if generator_name == 'MiniCPM-V2.6':
        model, tokenizer = get_minicpmv2_6(generator_name)
        response = generate_response(query, topk_image_paths, model, tokenizer)
    elif generator_name == 'gemini-2.5-pro':
        client = get_client()
        response = generate_gemini_response(query, topk_image_paths, client, generator_name)
    return response

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Multi-Modal RAG Demo", layout="wide")
st.title("📄 Multi-modal Document RAG")

st.sidebar.header("模型配置")
# 用于选择框
retriever_options = list(RAG_CONFIG.retrievers.keys())
generator_options = list(RAG_CONFIG.generators.keys())

retriever = st.sidebar.selectbox("选择检索模型", retriever_options)
generator = st.sidebar.selectbox("选择生成模型", generator_options)
granularity = st.sidebar.radio("检索粒度", ["page-level", "layout-level"])

uploaded_file = st.file_uploader("上传文档", type=["pdf", "png", "jpg"])

# 初始化 session state
if "index_map" not in st.session_state:
    st.session_state.index_map = {}  # 结构: {pdf_id: {granularity: {retriever: index_path}}}
if "image_map" not in st.session_state:
    st.session_state.image_map = {}  # 结构: {pdf_id: {granularity: image_dir}}

if uploaded_file:
    pdf_id = get_stable_pdf_id(uploaded_file)
    # 如果图像缓存已存在
    if pdf_id in st.session_state.image_map and granularity in st.session_state.image_map[pdf_id]:
        image_path = st.session_state.image_map[pdf_id][granularity]
        st.info(f"📄 使用已缓存图像路径（粒度: {granularity}）")
    else:
        with st.spinner("🔍 正在解析文档..."):
            image_path = document_parser(uploaded_file, granularity)
            st.session_state.image_map.setdefault(pdf_id, {})[granularity] = image_path
        st.success("✅ 图像已解析")

    # 判断索引是否已存在
    index_exists = (
            pdf_id in st.session_state.index_map and
            granularity in st.session_state.index_map[pdf_id] and
            retriever in st.session_state.index_map[pdf_id][granularity]
    )

    if not index_exists:
        with st.spinner("🔧 正在构建索引..."):
            index_path = build_index(image_path, retriever, granularity)
            st.session_state.index_map.setdefault(pdf_id, {})[granularity] = {}
            st.session_state.index_map[pdf_id][granularity][retriever] = index_path
        st.success("✅ 新索引已构建")
    else:
        index_path = st.session_state.index_map[pdf_id][granularity][retriever]
        st.info(f"📁 使用已缓存索引（Retriever: {retriever}）")

    query = st.text_input("请输入你的问题：")
    generate_flag = st.checkbox("是否生成回答", value=True)

    if query:
        with st.spinner("检索中..."):
            topk_image_paths = retrieve(query, retriever, index_path)

        st.subheader("🔍 检索到的内容")
        cols = st.columns(len(topk_image_paths))  # 每张图一个列
        for col, img_path in zip(cols, topk_image_paths):
            image = Image.open(img_path)
            col.image(image, use_container_width=True)

        if generate_flag:
            with st.spinner("生成回答中..."):
                response = generate_answer(topk_image_paths, query, generator)

            st.subheader("🤖 回答")
            st.write(response)
        else:
            st.info("🧠 当前未启用回答生成功能，仅展示检索结果。")
else:
    st.info("请在左侧上传文档以开始")
