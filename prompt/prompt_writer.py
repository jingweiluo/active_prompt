import numpy as np # type: ignore

def prompt4(train_data, train_label, test_data):
    with open("output_with_text.txt", "w", encoding='utf-8') as file:
        # Write task description
        file.write("### 请根据给出的示例，判断另外若干段EEG信号反映了被试的以下哪种情绪\n")
        # file.write("### 0表示中立，1表示悲伤，2表示恐惧，3表示高兴\n\n")
        file.write("      0: 中立\n")
        file.write("      1: 悲伤\n")
        file.write("      2: 恐惧\n")
        file.write("      3: 开心\n\n")
        # file.write("### 从以上四个选项中选择一个选项对应的数字作为该段信号的标签\n\n")

        # Write predicted EEG signal information
        # file.write("### 待预测的EEG信号的微分熵值如下，长度为170s，数组中的五个值分别对应如下5个频段的微分熵：\n")
        # file.write("      1. Delta: 1~4 Hz\n")
        # file.write("      2. Theta: 4~8 Hz\n")
        # file.write("      3. Alpha: 8~14 Hz\n")
        # file.write("      4. Beta: 14~31 Hz\n")
        # file.write("      5. Gamma: 31~50 Hz\n\n")
        emotionmap = {
            0: '中立',
            1: '悲伤',
            2: '恐惧',
            3: '开心'
        }

        # Write test_data content, assuming each element represents a channel's data
        channel_names = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]

        # Write example data
        # file.write("针对此问题有如下示例可做参考：\n")
        for (i, array_2d) in enumerate(train_data):
            file.write(f"### EEG信号样本{i+1}:\n")
            for (j, channel_name) in enumerate(channel_names):
                file.write(f"      在通道{channel_name}，Delta波段的平均微分熵为：{np.array2string(array_2d[j][0])}\n")
                file.write(f"      在通道{channel_name}，Theta波段的平均微分熵为：{np.array2string(array_2d[j][1])}\n")
                file.write(f"      在通道{channel_name}，Alpha波段的平均微分熵为：{np.array2string(array_2d[j][2])}\n")
                file.write(f"      在通道{channel_name}，Beta波段的平均微分熵为：{np.array2string(array_2d[j][3])}\n")
                file.write(f"      在通道{channel_name}，Gamma波段的平均微分熵为：{np.array2string(array_2d[j][4])}\n")
            # file.write(f"### 示例样本{i+1}的标签（label）: {train_label[i]}\n\n")
            file.write(f"### 它反映了被试者当前的情绪为： {emotionmap.get(train_label[i])}\n\n")

        file.write(f"### 请判断下面样本分别反应了被试者的什么情绪，请从'中立'，'悲伤'， '恐惧'，'开心'中选择一个，以列表的形式返回。\n\n")
        for (i, array_2d) in enumerate(test_data):
            file.write(f"### 待预测样本{i+1}:\n")
            for (j, channel_name) in enumerate(channel_names):
                file.write(f"      在通道{channel_name}，Delta波段的平均微分熵为：{np.array2string(array_2d[j][0])}\n")
                file.write(f"      在通道{channel_name}，Theta波段的平均微分熵为：{np.array2string(array_2d[j][1])}\n")
                file.write(f"      在通道{channel_name}，Alpha波段的平均微分熵为：{np.array2string(array_2d[j][2])}\n")
                file.write(f"      在通道{channel_name}，Beta波段的平均微分熵为：{np.array2string(array_2d[j][3])}\n")
                file.write(f"      在通道{channel_name}，Gamma波段的平均微分熵为：{np.array2string(array_2d[j][4])}\n")
            file.write("\n")
        file.write("\n")

        # JSON format final result placeholder
        file.write("### 请以python列表的json格式返回待预测样本分别对应的标签数字，标签值为'中立'返回0，'悲伤'返回1， '恐惧'返回2，'开心'返回3。\n")
        file.write("### 只返回json列表，不要返回任何多余内容")
        # file.write("### 请输出最可能的、确定性最高、概率最大的结果\n")
        # file.write("### 请基于每个样本的特征，输出概率最高、最确定的标签结果，忽略任何不确定性因素，专注于每个情绪类别中微分熵特征的最大可能性。\n")


def prompt_emotion(train_data, train_label, test_data):
    emotionmap = {
        0: '中立',
        1: '悲伤',
        2: '恐惧',
        3: '开心'
    }
    channel_names = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]

    with open("output_with_text.txt", "w", encoding='utf-8') as file:
        file.write("### 请根据给出的示例,判断待预测的EEG信号反映了被试的以下哪种情绪\n")
        file.write("### 0表示中立,1表示悲伤,2表示恐惧,3表示开心\n\n")

        for (i, array_2d) in enumerate(train_data):
            file.write(f"### EEG信号样本{i+1}:\n")
            for (j, channel_name) in enumerate(channel_names):
                file.write(f"      在通道{channel_name},五个波段的平均微分熵为:Delta:{np.array2string(array_2d[j][0])},Theta:{np.array2string(array_2d[j][1])},Alpha:{np.array2string(array_2d[j][2])},Beta:{np.array2string(array_2d[j][3])},Gamma:{np.array2string(array_2d[j][4])},\n")
            file.write(f"### 它反映了被试者当前的情绪为： {emotionmap.get(train_label[i])}\n\n")

        file.write(f"### 请判断下面样本分别反应了被试者的什么情绪，请从'中立'，'悲伤'， '恐惧'，'开心'中选择一个，以列表的形式返回。\n\n")
        for (i, array_2d) in enumerate(test_data):
            file.write(f"### 待预测样本{i+1}:\n")
            for (j, channel_name) in enumerate(channel_names):
                file.write(f"      在通道{channel_name},五个波段的平均微分熵为:Delta:{np.array2string(array_2d[j][0])},Theta:{np.array2string(array_2d[j][1])},Alpha:{np.array2string(array_2d[j][2])},Beta:{np.array2string(array_2d[j][3])},Gamma:{np.array2string(array_2d[j][4])},\n")
            file.write("\n")
        file.write("\n")

        # JSON format final result placeholder
        file.write("### 请按照以下要求返回待预测样本对应的标签：\n")
        file.write("### 1. 返回的结果必须是一个 Python 列表，并以 JSON 格式表示。\n")
        file.write("### 2. 列表中的每个元素均为数字，具体对应关系如下：\n")
        file.write("###    - '中立' 对应标签值为 0\n")
        file.write("###    - '悲伤' 对应标签值为 1\n")
        file.write("###    - '恐惧' 对应标签值为 2\n")
        file.write("###    - '开心' 对应标签值为 3\n")
        file.write(f"### 3. 列表的长度必须为{len(test_data)}(与待预测样本的数量一致)。\n")
        file.write("### 4. 仅返回标签列表，不包含任何多余内容。\n")

def prompt_back(train_data, train_label, test_data):
    with open("output_with_text.txt", "w", encoding='utf-8') as file:
        file.write("### 请根据给出的示例,判断待预测的EEG信号反映了被试在想象左手(left_hand)还是右手(right_hand)的运动\n")
        for (i, array_2d) in enumerate(train_data):
            file.write(f"### EEG信号样本{i+1}:\n")
            file.write(f"### csp特征值为{np.array2string(array_2d.flatten(), separator=' ')}\n")
            file.write(f"### 它反映了被试者当前在想象的运动是： {train_label[i]}\n\n")

        file.write(f"### 请判断下面样本分别反应了被试者在想象的运动是什么，请从'left_hand', 'right_hand'中选择一个，以列表的形式返回。\n\n")
        for (i, array_2d) in enumerate(test_data):
            file.write(f"### 待预测样本{i+1}:\n")
            file.write(f"### csp特征值为{np.array2string(array_2d.flatten(), separator=' ')}\n\n")
        file.write("\n")

        # JSON format final result placeholder
        file.write("### 请按照以下要求返回待预测样本对应的标签：\n")
        file.write("### 1. 返回的结果必须是一个 Python 列表，并以 JSON 格式表示。\n")
        file.write("### 2. 列表中的每个元素均为字符，'left_hand', 'right_hand'\n")
        file.write(f"### 3. 列表的长度必须为{len(test_data)}(与待预测样本的数量一致)。\n")
        file.write("### 4. 仅返回标签列表，不包含任何多余内容。\n")

def prompt(train_data, train_label, test_data):
    # 开始写文件
    with open("output_with_text.txt", "w", encoding='utf-8') as file:
        # 任务描述
        file.write("### 任务描述 ###\n")
        file.write("给定一组脑电信号（EEG）样本的特征值，判断待预测样本所对应的想象运动是左手（left_hand）还是右手（right_hand）。\n\n")

        # # 背景信息
        # file.write("### 背景信息 ###\n")
        # file.write("样本的特征值是从EEG数据中提取的，得到的过程如下：首先将原始EEG信号在8-32Hz的频率范围内分解成6个频段，每隔4Hz一个频段。接着，对每个频段的信号应用共空间模式（CSP）算法，从中提取8个最重要的主成分。这48个特征值（每个频段8个）合并形成了这个一维信号数组，有效捕捉了脑电活动在不同频段下的空间分布和变化情况，适用于进一步的分析和模式识别任务。\n\n")

        # 分析方法
        file.write("### 分析方法 ###\n")
        file.write("让我们一步步分析\n\n")
        # file.write("让我们一步步分析，请构建一个分类模型来分类EEG数据 \n\n")

        # 示例样本
        file.write("### 示例样本 ###\n")
        for i, array_2d in enumerate(train_data):
            file.write(f"#### 样本 {i+1} ####\n")
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n")
            file.write(f"标签: {train_label[i]}\n\n")

        # 待预测样本
        file.write("### 待预测样本 ###\n")
        for i, array_2d in enumerate(test_data):
            file.write(f"#### 样本 {i+1} ####\n")
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n\n")

        # 结果要求
        file.write("### 要求 ###\n")
        # file.write("1. 分析每个样本的CSP特征值，确定被试者在想象的是左手还是右手。\n")
        file.write("1. 返回的结果必须是一个 Python列表，并以 JSON 格式表示，所有字符串必须用双引号表示。\n")
        file.write("2. 列表中的每个元素均为字符，'left_hand', 'right_hand'\n")
        file.write(f"3. 列表的长度必须为{len(test_data)}(与待预测样本的数量一致)。\n")
        file.write("4. 仅返回标签列表，请不要包含任何多余内容！\n")

def prompt_dynamic(left_demo, right_demo, train_data, train_label, test_data):
    # 开始写文件
    with open("output_with_text.txt", "w", encoding='utf-8') as file:
        # 任务描述
        file.write("### 任务描述 ###\n")
        file.write("给定一组脑电信号（EEG）样本的特征值，判断待预测样本所对应的想象运动是左手（left_hand）还是右手（right_hand）。\n\n")

        # # 背景信息
        # file.write("### 背景信息 ###\n")
        # file.write("样本的特征值是从EEG数据中提取的，得到的过程如下：首先将原始EEG信号在8-32Hz的频率范围内分解成6个频段，每隔4Hz一个频段。接着，对每个频段的信号应用共空间模式（CSP）算法，从中提取8个最重要的主成分。这48个特征值（每个频段8个）合并形成了这个一维信号数组，有效捕捉了脑电活动在不同频段下的空间分布和变化情况，适用于进一步的分析和模式识别任务。\n\n")

        # 分析方法
        file.write("### 分析方法 ###\n")
        file.write("让我们一步步分析 \n\n")
        # file.write("让我们一步步分析，请构建一个分类模型来分类EEG数据 \n\n")

        file.write("### 左手运动想象示例样本 ###\n")
        for i, array_2d in enumerate(left_demo):
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n")
            file.write(f"标签: 'left_hand'\n\n")

        file.write("### 右手运动想象示例样本 ###\n")
        for i, array_2d in enumerate(right_demo):
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n")
            file.write(f"标签: 'right_hand'\n\n")

        # 示例样本
        file.write("### 以下是与待预测样本距离最接近的示例样本 ###\n")
        for i, array_2d in enumerate(train_data):
            file.write(f"#### 样本 {i+1} ####\n")
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n")
            if i < 1:
                file.write(f"标签: {train_label[i]}\n\n")

        # 待预测样本
        file.write("### 待预测样本 ###\n")
        for i, array_2d in enumerate(test_data):
            file.write(f"#### 样本 {i+1} ####\n")
            file.write(f"特征值: {np.array2string(array_2d.flatten(), separator=' ')}\n\n")

        # 结果要求
        file.write("### 要求 ###\n")
        # file.write("1. 分析每个样本的CSP特征值，确定被试者在想象的是左手还是右手。\n")
        file.write("1. 返回的结果必须是一个 Python列表，并以 JSON 格式表示，所有字符串必须用双引号表示。\n")
        file.write("2. 列表中的每个元素均为字符，'left_hand', 'right_hand'\n")
        file.write(f"3. 列表的长度必须为{len(test_data)}(与待预测样本的数量一致)。\n")
        file.write("4. 仅返回标签列表，请不要包含任何多余内容！\n")