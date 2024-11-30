from openai import OpenAI # type: ignore
import os
from dotenv import load_dotenv # type: ignore
load_dotenv()


client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义函数来调用 ChatGPT
def ask_llm(model_type):
    with open("output_with_text.txt", "r", encoding="utf-8") as file:
        prompt = file.read()

    response = client.chat.completions.create(
        model= model_type,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            # {
            #     "role": "system",
            #     "content": "You are an expert in data science and machine learning. Your role is to analyze high-dimensional data and uncover hidden patterns or correlations. Focus on tasks such as identifying clusters, key features, or trends in the data. If necessary, propose dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP) or clustering methods (e.g., K-means, DBSCAN). Explain your findings in a way that is interpretable to someone with a basic understanding of data science."
            # },
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # 设置温度较低以提高回答的一致性
        max_tokens=8192  # 设置回答的最大 token 数量
    )
    # 获取并返回模型生成的文本
    return response.choices[0].message.content
