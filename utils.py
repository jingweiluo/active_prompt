import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore
from collections import Counter
from active_prompt.prompt.llm import ask_llm
from active_prompt.prompt.prompt_writer import prompt
import datetime

def extract_array_from_string(input_string):
    """
    从字符串中提取合法的 JSON 数组内容。
    
    参数:
    - data: str，包含 JSON 数据的字符串
    
    返回:
    - list，提取出的数组内容
    """
    try:
        # 使用正则表达式提取数组部分
        match = re.search(r'\[.*?\]', input_string)
        if match:
            json_array = match.group(0)  # 提取匹配的数组部分
            return json.loads(json_array)  # 将数组部分解析为 Python 列表
        else:
            raise ValueError("未找到合法的 JSON 数组")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解码失败: {e}")

def get_accuracy_and_log(y_true, y_pred, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices, selected_labels):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # 计算精确度
    precision = precision_score(y_true, y_pred, average='macro')  # 'macro' 表示未加权的均值
    print(f"Precision: {precision}")

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')
    print(f"Recall: {recall}")

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score: {f1}")

    # 记录到日志
    with open(f"log/result_log_{way_select_demo}", 'a',  encoding='utf-8') as file:
        # 记录时间戳、实验名称和结果
        file.write(f"======================================================================\n")
        file.write(f"--- {datetime.datetime.now()} ---\n")
        file.write(f"test_mode: 测试集(inner表示80%中除去demo, outer表示20%的测试集){test_mode}\n")
        file.write(f"model_type: {model_type}\n")
        file.write(f"sub_index: {sub_index}\n")
        file.write(f"num_demos: {num_demos}\n")
        file.write(f"max_predict_num: {max_predict_num}\n")
        file.write(f"选择演示示例的方式: {way_select_demo}\n")
        file.write(f"选择的index: {selected_indices}\n")
        file.write(f"选择的labels: {selected_labels}\n")

        file.write(f"True Label: {y_true}\n")
        file.write(f"Pred Label: {y_pred}\n")

        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

        file.write("\n")  # 添加换行，便于区分记录

def most_common_element(lst):
    """
    返回列表中出现次数最多的元素
    :param lst: 输入列表
    :return: 出现次数最多的元素及其次数
    """
    if not lst:
        return None, 0  # 处理空列表情况
    
    count = Counter(lst)
    most_common = count.most_common(1)[0]  # 获取出现次数最多的元素
    return most_common[0], most_common[1]

def collect_y_pred(demo_data, demo_labels, predict_data, max_predict_num, model_type):
    """
    将多次返回的lst汇总为一个lst
    demo_data: 用来做演示示例的4个trial,lst of ndArray
    demo_labels: 4个trial的label, lst
    predict_data: 待遇测的trial lst
    max_predict_num: 单次最多预测条数
    """
    y_pred = []
    for i in range (0, len(predict_data), max_predict_num):
        test_data = predict_data[i:i+max_predict_num] if i + max_predict_num <= len(predict_data) else predict_data[i:len(predict_data)]
        prompt(demo_data, demo_labels, test_data)
        answer = ask_llm(model_type)
        print(f'第{(i // max_predict_num) + 1}段,API输出为:{answer}')
        y_pred_sub = extract_array_from_string(answer)
        if y_pred_sub:  # 确保提取的结果不为空
            y_pred.extend(y_pred_sub)
    return y_pred