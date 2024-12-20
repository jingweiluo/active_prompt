# from openai import OpenAI # type: ignore
# import os
# from dotenv import load_dotenv # type: ignore
# load_dotenv()


# client = OpenAI(
#     api_key=os.getenv("QWEN_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# # 定义函数来调用 ChatGPT
# def ask_llm(model_type):
#     with open("output_with_text.txt", "r", encoding="utf-8") as file:
#         prompt = file.read()

#     response = client.chat.completions.create(
#         model= model_type,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             # {
#             #     "role": "system",
#             #     "content": "You are an expert in data science and machine learning. Your role is to analyze high-dimensional data and uncover hidden patterns or correlations. Focus on tasks such as identifying clusters, key features, or trends in the data. If necessary, propose dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP) or clustering methods (e.g., K-means, DBSCAN). Explain your findings in a way that is interpretable to someone with a basic understanding of data science."
#             # },
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0,  # 设置温度较低以提高回答的一致性
#         max_tokens=8192  # 设置回答的最大 token 数量
#     )
#     # 获取并返回模型生成的文本
#     return response.choices[0].message.content

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量，缓存模型和 Tokenizer
_loaded_model = None
_loaded_tokenizer = None

def ask_llm(model_type):
    global _loaded_model, _loaded_tokenizer

    # 如果模型和 Tokenizer 尚未加载，则加载
    if _loaded_model is None or _loaded_tokenizer is None:
        _loaded_model = AutoModelForCausalLM.from_pretrained(
            model_type,
            torch_dtype="auto",
            device_map="auto",
            # force_download=True
        )
        _loaded_tokenizer = AutoTokenizer.from_pretrained(model_type)
        print("Model and tokenizer loaded.")

    # 使用已加载的模型和 Tokenizer
    model = _loaded_model
    tokenizer = _loaded_tokenizer

    with open("output_with_text.txt", "r", encoding="utf-8") as file:
        prompt = file.read()

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.2  # 控制生成的随机性
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct Meta-Llama-3-8B-Instruct
# ask_llm("Qwen/Qwen2.5-7B-Instruct")
