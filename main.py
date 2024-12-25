import numpy as np # type: ignore
import random
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(current_file_path))
print(parent_parent_dir)
sys.path.append(parent_parent_dir)
from active_prompt.prompt.prompt_writer import prompt
from active_prompt.query.active_select import BasicRD, find_k_similar, find_k_similar_for_each_label
from utils import extract_array_from_string, collect_y_pred, get_accuracy_and_log
from active_prompt.load.load_moabb import get_moabb_data
from active_prompt.query.qbc import qbc
from tqdm import tqdm
# from llm import ask_llm

def get_active_learned_samples_indices(train_data, num_demos):
    train_arr = np.array(train_data)
    train_arr = train_arr.reshape(train_arr.shape[0], -1)

    RD_instance = BasicRD(num_demos // 2, num_demos)
    selected_samples, selected_indices = RD_instance.rd_iterate(train_arr, selection_strategy="basic")
    return selected_indices

def ndToList(array):
    return [array[i] for i in range(array.shape[0])]

def active_learning_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, is_model_online):
    if way_select_demo == "random":
        selected_indices = random.sample(list(range(len(train_labels))), num_demos)
    elif way_select_demo == "basic_rd":
        selected_indices = get_active_learned_samples_indices(train_data, num_demos)
    elif way_select_demo == "qbc":
        selected_indices,_ = qbc(train_data, train_labels, n_members=5, n_initial=12, n_queries=num_demos)
    selected_labels = [train_labels[i] for i in selected_indices]

    demo_data = [train_data[i] for i in selected_indices]
    demo_labels = [train_labels[i] for i in selected_indices]
    sub_test_data = [train_data[i] for i in range(len(train_data)) if i not in selected_indices]
    sub_test_labels = [train_labels[i] for i in range(len(train_labels)) if i not in selected_indices]

    if test_mode == 'inner_test':
        y_true = sub_test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, sub_test_data, max_predict_num, model_type, is_model_online)
    elif test_mode == 'outer_test':
        y_true = test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, test_data, max_predict_num, model_type, is_model_online)
    accuracy, precision, recall, f1 = get_accuracy_and_log(y_true, y_pred, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices, selected_labels)
    return accuracy, precision, recall, f1

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
    sub_index = 3 # 被试编号(1-9)
    max_predict_num = 10 # 单次最多预测样本的个数，演示示例+单次预测样本个数，加起来的本文长度不能超过LLM的max_token
    model_type = 'qwen2.5-3b-instruct'# "qwen2.5-3b-instruct" "Qwen/Qwen2.5-Coder-32B-Instruct" # 'Qwen/Qwen2.5-7B-Instruct'# "qwen2.5-7b-instruct" Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    way_select_demo = "random" # basic_rd, random, qbc
    test_mode = "outer_test"
    dataset_name = "2a"
    test_id = 1 # 2a中只有0-1两个session，2b中有0-4五个session
    is_model_online = True # 谨慎开启，设置为online时要提前计算费用
    repeat_times = 1

    # kate_learning_predict(num_demos, sub_index)

    train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, sub_index, test_id) # 2b数据集，sub_index, test_id
    train_data = ndToList(train_data)
    test_data = ndToList(test_data)
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()

    def compare_test(n_times):
        qbc_score_list = []
        rand_score_list = []
        for i in tqdm(range(n_times)):
            accuracy, precision, recall, f1 = active_learning_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'qbc', is_model_online)
            accuracy2, precision2, recall2, f12 = active_learning_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'random', is_model_online)

            qbc_score_list.append(accuracy)
            rand_score_list.append(accuracy2)

            qbc_mean = np.mean(qbc_score_list)
            random_mean = np.mean(rand_score_list)
            print(qbc_mean, random_mean)
    
    compare_test(repeat_times)

