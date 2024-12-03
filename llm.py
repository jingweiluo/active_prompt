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

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

def ask_llm():
    with open("output_with_text.txt", "r", encoding="utf-8") as file:
        prompt = file.read()
    
        # 选择模型的ID或路径，可以是Hugging Face上的预训练模型，也可以是本地路径
        model_id = "gpt2"  # 替换为你需要的模型ID，例如 'gpt2' 或其他模型的ID 'meta-llama/Meta-Llama-3-8B-Instruct'

        # 1. 加载模型配置
        config = AutoConfig.from_pretrained(model_id)

        # 2. 加载模型
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config)

        # 3. 加载tokenizer（词元化器）
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 4. 使用模型进行推理
        # 输入文本
        input_text = prompt

        # 将输入文本转化为模型的输入格式
        inputs = tokenizer(input_text, return_tensors="pt")

        # 使用模型进行推理，得到输出
        with torch.no_grad():
            # 设置温度
            temperature = 0.2  # 可调节的温度参数，通常在 0.7 到 1.0 之间

            # 生成文本
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=temperature,  # 设置温度
                do_sample=True,           # 必须启用采样以使温度生效
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated text:", generated_text)

ask_llm()
