# import torch
# import GPUtil

# def query_gpu_status_with_gputil():
#     gpus = GPUtil.getGPUs()
#     for gpu in gpus:
#         print(f"GPU ID: {gpu.id}")
#         print(f"GPU Name: {gpu.name}")
#         print(f"Load: {gpu.load*100}%")
#         print(f"Free Memory: {gpu.memoryFree}MB")
#         print(f"Used Memory: {gpu.memoryUsed}MB")
#         print(f"Total Memory: {gpu.memoryTotal}MB")
#         print(f"Temperature: {gpu.temperature} °C")
#         print('------------------------')

# # 调用函数
# query_gpu_status_with_gputil()

# print(f'当前已分配显存: {torch.cuda.memory_allocated()} bytes')
# print(f'当前保留显存: {torch.cuda.memory_reserved()} bytes')
# print(f'最大已分配显存: {torch.cuda.max_memory_allocated()} bytes')
# print(f'最大保留显存: {torch.cuda.max_memory_reserved()} bytes')


