import numpy as np # type: ignore
import random

from prompt_writer import prompt
from active_select import BasicRD, find_k_similar, find_k_similar_for_each_label
from load_data import load_all_trials
from utils import extract_array_from_string, collect_y_pred, get_accuracy_and_log
# from llm import ask_llm

def get_active_learned_samples_indices(train_data, num_demos):
    train_arr = np.array(train_data)
    train_arr = train_arr.reshape(train_arr.shape[0], -1)

    RD_instance = BasicRD(num_demos // 2, num_demos)
    selected_samples, selected_indices = RD_instance.rd_iterate(train_arr, selection_strategy="basic")
    return selected_indices

def ndToList(array):
    return [array[i] for i in range(array.shape[0])]

def active_learning_predict(test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo):
    # train_data, test_data, train_labels, test_labels = load_all_trials(sub_index)

    moabb_2b_loaded = np.load('../mi_pipeline/moabb_bci_iv_2b.npz')
    train_data, test_data, train_labels, test_labels = (moabb_2b_loaded[key] for key in moabb_2b_loaded)
    train_data = ndToList(train_data)
    test_data = ndToList(test_data)
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()

    if way_select_demo == "random":
        selected_indices = random.sample(list(range(len(train_labels))), num_demos)
    elif way_select_demo == "basic_rd":
        selected_indices = get_active_learned_samples_indices(train_data, num_demos)
    selected_labels = [train_labels[i] for i in selected_indices]

    demo_data = [train_data[i] for i in selected_indices]
    demo_labels = [train_labels[i] for i in selected_indices]
    sub_test_data = [train_data[i] for i in range(len(train_data)) if i not in selected_indices]
    sub_test_labels = [train_labels[i] for i in range(len(train_labels)) if i not in selected_indices]

    if test_mode == 'inner_test':
        y_true = sub_test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, sub_test_data, max_predict_num, model_type)
    elif test_mode == 'outer_test':
        y_true = test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, test_data, max_predict_num, model_type)
    get_accuracy_and_log(y_true, y_pred, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices, selected_labels)


# def kate_learning_predict(k, sub_index):
#     train_data, test_data, train_labels, test_labels = load_all_trials(sub_index)

#     y_pred = []
#     for i in range(len(train_data)):
#         # selected_indices = find_k_similar(k, train_data[i], train_data) # select k samples with the cosine similarity closest to the sample to be predicted as example samples.
#         # selected_indices = find_k_similar_for_each_label(k, train_data[i], train_data, train_labels)
#         selected_indices = random.sample(list(range(len(train_data))),k) # randomly select k samples as example samples.

#         example_data = [train_data[i] for i in selected_indices]
#         example_labels = [train_labels[i] for i in selected_indices]
#         predict_data = [train_data[i]]

#         prompt(example_data, example_labels, predict_data)
#         answer = ask_llm()
#         y_pred_sub = extract_array_from_string(answer)
#         print(f'第{i + 1}个trial,API输出为:{answer} \n 预测结果为：\n', y_pred_sub, '\n')
#         if y_pred_sub:  # 确保提取的结果不为空
#             y_pred.extend(y_pred_sub)

#     print(len(y_pred), y_pred)
#     print(len(train_labels), train_labels)
#     get_accuracy_and_log(train_labels, y_pred)

if __name__ == '__main__':
    num_demos = 8 # 演示示例的数量
    sub_index = 1 # 被试编号(1-9)
    max_predict_num = 10 # 单次最多预测样本的个数，演示示例+单次预测样本个数，加起来的本文长度不能超过LLM的max_token
    model_type = "Qwen/Qwen2.5-7B-Instruct" # 'Qwen/Qwen2.5-7B-Instruct'# "qwen2.5-7b-instruct"
    way_select_demo = "basic_rd" # basic_rd, random

    active_learning_predict('outer_test', num_demos, sub_index, max_predict_num, model_type, way_select_demo)
    # kate_learning_predict(num_demos, sub_index)

