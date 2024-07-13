# 时间：2024/06/21
# 介绍：本代码用于将各工况下的数据合并到一起

# 用到的库
import os
import csv
import pandas as pd
import numpy as np

# 主函数
# 储存着待读取的文件目录信息
with open("folder.csv", "r") as file:
    reader1 = csv.reader(file)
    wind_speed_1 = next(reader1)[0:]  # 迎向风速
    wind_speed_2 = next(reader1)[0:]  # 横向风速

# 生成待读取文件目录列表
physical_quantity = "p_"  # 获取的物理量类型：压力系数
suffix_name = ".csv"  # 文件后缀名
folder_list = []
for speed1 in wind_speed_1:
    for speed2 in wind_speed_2:
        folder_list.append(f"../data_for_CS/{physical_quantity}{speed1}_{speed2}{suffix_name}")

# 初始化一个空的列表，用于存储每份文件的第一列数据
matrix_data = []

# 获取文件总数
total_files = len(folder_list)

# 遍历列表中的每一份文件
for index, file in enumerate(folder_list):
    df = pd.read_csv(file)  # 读取CSV文件
    # df.iloc[:, 0] 获取第一列数据，df[1:] 获取从第二行开始的所有行
    column_data = df[1:].iloc[:, 0]
    matrix_data.append(column_data)  # 将提取的数据添加到列表中
    print(f"处理进度：{index + 1}/{total_files} - 已处理文件：{file}")  # 打印进度信息

# 将列表转换为NumPy数组
matrix = np.array(matrix_data)
matrix = matrix.T
df_matrix = pd.DataFrame(matrix)

# 保存文件
output_file = f"data_all_1D.csv"  # 输出文件路径
df_matrix.to_csv(f"{output_file}", index=False, header=False)
