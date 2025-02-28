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
        left_local_indices = random.sample(list(range(len(left_indices))), num_demos // 2)
        right_local_indices = random.sample(list(range(len(right_indices))), num_demos // 2)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)
    elif way_select_demo == "basic_rd":
        selected_indices = get_active_learned_samples_indices(train_data, num_demos)
    elif way_select_demo == "qbc":
        selected_indices,_ = qbc(train_data, train_labels, n_members=5, n_initial=8, n_queries=num_demos)
    elif way_select_demo == "qbc_return_all":
        selected_indices,_ = qbc(train_data, train_labels, n_members=5, n_initial=4, n_queries=num_demos-4, is_return_all=True)
    elif way_select_demo == "rd_basic":
        left_init_local_indices = find_centroids(left_data, left_labels, 1, measurement)
        right_init_local_indices = find_centroids(right_data, right_labels, 1, measurement)
        left_local_indices,_ = rd("basic", left_data, left_labels, n_members=5, n_initial=1, n_queries=num_demos // 2, measurement=measurement, estimator=RandomForestClassifier, init_indices=left_init_local_indices)
        right_local_indices,_ = rd("basic", right_data, right_labels, n_members=5, n_initial=1, n_queries=num_demos // 2, measurement=measurement, estimator=RandomForestClassifier, init_indices=right_init_local_indices)
        print(left_local_indices, right_local_indices)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)
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

def vote_k(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, is_model_online, test_num, combine_method, way_anno, annotate_num):
    left_indices = [i for i,l in enumerate(train_labels) if l == 'left_hand']
    right_indices = [i for i,l in enumerate(train_labels) if l == 'right_hand']
    left_data = [train_data[i] for i in left_indices]
    right_data = [train_data[i] for i in right_indices]
    left_labels = [train_labels[i] for i in left_indices]
    right_labels = [train_labels[i] for i in right_indices]

    if way_anno == "random":
        left_local_indices = random.sample(list(range(len(left_indices))), annotate_num // 2)
        right_local_indices = random.sample(list(range(len(right_indices))), annotate_num // 2)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)
    elif way_anno == "rd_basic":
        selected_indices,_ = rd("basic", train_data, train_labels, n_members=5, n_initial=2, n_queries=8, measurement=measurement, estimator=RandomForestClassifier)
    elif way_anno == "centroids":
        left_local_indices = find_centroids(left_data, left_labels, annotate_num // 2, measurement)
        right_local_indices = find_centroids(right_data, right_labels, annotate_num // 2, measurement)
        selected_indices = combineIndices(combine_method, left_local_indices, right_local_indices, left_indices, right_indices)

    # 通过选中的索引列表，得到对应的标注数据和标注标签
    anno_data = [train_data[i] for i in selected_indices]
    anno_labels = [train_labels[i] for i in selected_indices]

    test_data = np.array(test_data)
    Y = test_data.reshape(test_data.shape[0],-1)

    y_pred_list = []
    for i in range(Y.shape[0]):
        if way_select_demo == "random":
            local_indices = random.sample(list(range(len(anno_labels))), num_demos)
        elif way_select_demo == "kate_basic":
            local_indices = kate_basic(anno_data, anno_labels, Y[i], num_demos, measurement)

        example_data = [anno_data[i] for i in local_indices]
        example_labels = [anno_labels[i] for i in local_indices]
        predict_data = [test_data[i]]

        # y_pred = collect_y_pred_single(example_data, example_labels, predict_data, model_type, is_model_online)
        y_pred = [find_most_common(example_labels)[0]] # knn算法
        y_pred_list.extend(y_pred)

    accuracy, precision, recall, f1 = get_accuracy_and_log(test_labels, y_pred_list, test_mode, num_demos, sub_index, max_predict_num, model_type, way_select_demo, selected_indices=[], selected_labels=[])
    return accuracy, precision, recall, f1

from collections import Counter

def find_most_common(lst):
    count = Counter(lst)
    max_count = max(count.values())
    most_common_elements = [element for element, cnt in count.items() if cnt == max_count]
    return most_common_elements

if __name__ == '__main__':
    annotate_num = 4
    num_demos = 4 # 演示示例的数量
    sub_index = 3 # 被试编号(1-9)
    max_predict_num = 10 # 单次最多预测样本的个数，演示示例+单次预测样本个数，加起来的本文长度不能超过LLM的max_token
    model_type = "qwen2.5-14b-instruct-1m" # "moonshot-v1-32k", "deepseek-chat", "deepseek-reasoner", "qwen-long" "qwen2.5-3b-instruct" "Qwen/Qwen2.5-Coder-32B-Instruct" # 'Qwen/Qwen2.5-7B-Instruct'# "qwen2.5-7b-instruct" Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    # way_select_demo = "random" # basic_rd, random, qbc
    test_mode = "outer_test"
    dataset_name = "2a"
    test_id = 1 # 2a中只有0-1两个session，2b中有0-4五个session
    is_model_online = True # 谨慎开启，设置为online时要提前计算费用
    measurement = 'ed' # 衡量向量距离的方式，ed, cs
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
        list7 = []
        list8 = []

        for i in tqdm(range(n_times)):
            accuracy, precision, recall, f1 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'random', is_model_online, test_num, combine_method)
            accuracy2, precision2, recall2, f12 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'cetroids', is_model_online, test_num, combine_method)
            # accuracy3, precision3, recall3, f13 = static_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, 'cetroids', is_model_online, test_num, combine_method)
            # accuracy4, precision4, recall4, f14 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_basic", is_model_online, measurement)
            # accuracy5, precision5, recall5, f15 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "find_centroids", is_model_online, measurement)
            # accuracy6, precision6, recall6, f16 = dynamic_demo_predict(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_qbc", is_model_online, measurement)
            # accuracy7, precision7, recall7, f17 = vote_k(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_basic", is_model_online, test_num, combine_method, "random", annotate_num)
            # accuracy8, precision8, recall8, f18 = vote_k(train_data, test_data, train_labels, test_labels, test_mode, num_demos, sub_index, max_predict_num, model_type, "kate_basic", is_model_online, test_num, combine_method, "centroids", annotate_num)

            list1.append(accuracy)
            list2.append(accuracy2)
            # list3.append(accuracy3)
            # list4.append(accuracy4)
            # list5.append(accuracy5)
            # list6.append(accuracy6)
            # list7.append(accuracy7)
            # list8.append(accuracy8)


            mean1 = np.mean(list1)
            std1 = np.std(list1)

            mean2 = np.mean(list2)
            std2 = np.std(list2)

            print(mean1, std1, mean2, std2)

    compare_test(repeat_times)

