import numpy as np # type: ignore
import random
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_parent_dir)
from active_prompt.prompt.prompt_writer import prompt
from active_prompt.query.active_select import BasicRD, find_k_similar, find_k_similar_for_each_label
from utils import collect_y_pred, get_accuracy_and_log, collect_y_pred_single
from active_prompt.load.load_moabb import get_moabb_data
from active_prompt.query.qbc import qbc, kate, rd, find_centroids, kate_qbc, kate_basic
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from llm import ask_llm

def get_active_learned_samples_indices(train_data, num_demos):
    train_arr = np.array(train_data)
    train_arr = train_arr.reshape(train_arr.shape[0], -1)

    RD_instance = BasicRD(num_demos // 2, num_demos)
    selected_samples, selected_indices = RD_instance.rd_iterate(train_arr, selection_strategy="basic")
    return selected_indices

def ndToList(array):
    return [array[i] for i in range(array.shape[0])]

# 将两类样本根据指定方式结合，返回一个index list
def combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices):
    left_selects_indices = [left_indices[i] for i in left_local_indices]
    right_selects_indices = [right_indices[i] for i in right_local_indices]
    if combine_method == 'shuffle':
        merge_list = left_selects_indices + right_selects_indices
        random.shuffle(merge_list)
        return merge_list
    elif combine_method == 'lasagna-left':
        merge_list = []
        for i in range(len(left_selects_indices)):
            merge_list.append(left_selects_indices[i])
            merge_list.append(right_selects_indices[i])
        return merge_list
    elif combine_method == 'lasagna-right':
        merge_list = []
        for i in range(len(left_selects_indices)):
            merge_list.append(right_selects_indices[i])
            merge_list.append(left_selects_indices[i])
        return merge_list



def static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, is_model_online, test_num, combine_method):
    left_indices = [i for i,l in enumerate(train_labels) if l == 'left_hand']
    right_indices = [i for i,l in enumerate(train_labels) if l == 'right_hand']
    left_data = [train_data[i] for i in left_indices]
    right_data = [train_data[i] for i in right_indices]
    left_labels = [train_labels[i] for i in left_indices]
    right_labels = [train_labels[i] for i in right_indices]

    if way_select_demo == "random":
        # selected_indices = random.sample(list(range(len(train_labels))), num_demos)
        left_local_indices = random.sample(list(range(len(left_indices))), num_demos//2)
        right_local_indices = random.sample(list(range(len(right_indices))), num_demos//2)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)
    elif way_select_demo == "basic_rd":
        selected_indices = get_active_learned_samples_indices(train_data, num_demos)
    elif way_select_demo == "qbc":
        selected_indices,_ = qbc(train_data, train_labels, n_members=5, n_initial=8, n_queries=num_demos)
    elif way_select_demo == "qbc_return_all":
        selected_indices,_ = qbc(train_data, train_labels, n_members=5, n_initial=4, n_queries=num_demos-4, is_return_all=True)
    elif way_select_demo == "rd_basic":
        selected_indices,_ = rd("basic", train_data, train_labels, n_members=5, n_initial=2, n_queries=8, measurement=measurement, estimator=RandomForestClassifier)
    elif way_select_demo == "rd_qbc":
        selected_indices,_ = rd("qbc", train_data, train_labels, n_members=5, n_initial=2, n_queries=8, measurement=measurement, estimator=RandomForestClassifier)
    elif way_select_demo == "cetroids":
        left_local_indices = find_centroids(left_data, left_labels, num_demos // 2, measurement)
        right_local_indices = find_centroids(right_data, right_labels, num_demos // 2, measurement)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)
    selected_labels = [train_labels[i] for i in selected_indices]

    demo_data = [train_data[i] for i in selected_indices]
    demo_labels = [train_labels[i] for i in selected_indices]
    sub_test_data = [train_data[i] for i in range(len(train_data)) if i not in selected_indices]
    sub_test_labels = [train_labels[i] for i in range(len(train_labels)) if i not in selected_indices]

    if test_mode == 'inner_test':
        y_true = sub_test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, sub_test_data, max_predict_num, model_type, is_model_online)
    elif test_mode == 'outer_test':
        if test_num:
            y_true = test_labels[:test_num]
            y_pred = collect_y_pred(demo_data, demo_labels, test_data[:test_num], max_predict_num, model_type, is_model_online)
        else:
            y_true = test_labels
            y_pred = collect_y_pred(demo_data, demo_labels, test_data, max_predict_num, model_type, is_model_online)
    print('真实标签：', y_true)
    accuracy, precision, recall, f1 = get_accuracy_and_log(y_true, y_pred, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices, selected_labels)
    return accuracy, precision, recall, f1

def dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, is_model_online, measurement):
    left_indices = [i for i,l in enumerate(train_labels) if l == 'left_hand']
    right_indices = [i for i,l in enumerate(train_labels) if l == 'right_hand']
    left_data = [train_data[i] for i in left_indices]
    right_data = [train_data[i] for i in right_indices]
    left_labels = [train_labels[i] for i in left_indices]
    right_labels = [train_labels[i] for i in right_indices]

    test_data = np.array(test_data)
    Y = test_data.reshape(test_data.shape[0],-1)

    y_pred_list = []
    for i in range(Y.shape[0]):
        if way_select_demo == "random":
            selected_indices = random.sample(list(range(len(train_labels))), num_demos)
        elif way_select_demo == "find_centroids":
            selected_indices = find_centroids(train_data, train_labels, num_demos, measurement)
        elif way_select_demo == "kate":
            selected_indices = kate(train_data, train_labels, Y[i], num_demos, measurement)
        elif way_select_demo == "kate_qbc":
            selected_indices,_ = kate_qbc(train_data, train_labels, Y[i], num_demos, measurement)
        elif way_select_demo == "kate_basic":
            left_local_indices = kate_basic(left_data, left_labels, Y[i], num_demos // 2, measurement)
            right_local_indices = kate_basic(right_data, right_labels, Y[i], num_demos // 2, measurement)
            selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)

        example_data = [train_data[i] for i in selected_indices]
        example_labels = [train_labels[i] for i in selected_indices]
        predict_data = [test_data[i]]

        y_pred = collect_y_pred_single(example_data, example_labels, predict_data, model_type, is_model_online)
        y_pred_list.extend(y_pred)

    accuracy, precision, recall, f1 = get_accuracy_and_log(test_labels, y_pred_list, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices=[], selected_labels=[])
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    num_demos = 8 # 演示示例的数量
    sub_index = 3 # 被试编号(1-9)
    max_predict_num = 10 # 单次最多预测样本的个数，演示示例+单次预测样本个数，加起来的本文长度不能超过LLM的max_token
    model_type = "qwen2.5-14b-instruct-1m" # "moonshot-v1-32k", "deepseek-chat", "deepseek-reasoner", "qwen-long" "qwen2.5-3b-instruct" "Qwen/Qwen2.5-Coder-32B-Instruct" # 'Qwen/Qwen2.5-7B-Instruct'# "qwen2.5-7b-instruct" Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    # way_select_demo = "random" # basic_rd, random, qbc
    test_mode = "outer_test"
    dataset_name = "2a"
    test_id = 1 # 2a中只有0-1两个session，2b中有0-4五个session
    is_model_online = True # 谨慎开启，设置为online时要提前计算费用
    measurement = 'cs' # 衡量向量距离的方式，ed, cs
    combine_method = 'shuffle' # shuffle, lasagna-left, lasagna-right
    repeat_times = 30
    test_num = None # 10, None

    train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, sub_index, test_id) # 2b数据集，sub_index, test_id
    train_data = ndToList(train_data)
    test_data = ndToList(test_data)
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()

    def compare_test(n_times):
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        list5 = []
        list6 = []


        for i in tqdm(range(n_times)):
            # accuracy, precision, recall, f1 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'random', is_model_online, test_num, combine_method)
            # accuracy2, precision2, recall2, f12 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'rd', is_model_online, test_num, combine_method)
            # accuracy3, precision3, recall3, f13 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'cetroids', is_model_online, test_num, combine_method)
            accuracy4, precision4, recall4, f14 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_basic", is_model_online, measurement)
            # accuracy5, precision5, recall5, f15 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "find_centroids", is_model_online, measurement)
            # accuracy6, precision6, recall6, f16 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_qbc", is_model_online, measurement)

            # list1.append(accuracy)
            # list1.append(accuracy2)
            # list3.append(accuracy3)
            list4.append(accuracy4)
            # list5.append(accuracy5)
            # list6.append(accuracy6)

            mean1 = np.mean(list4)
            std1 = np.std(list4)

            # mean3 = np.mean(list3)
            # std3 = np.std(list3)

            print(mean1, std1)

    compare_test(repeat_times)

