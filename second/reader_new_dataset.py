import chardet as chardet
import networkx as nx
import pickle

# 读取数据并添加边
# file_path = r"C:\Users\11792\Desktop\arXiv\2_following.txt"  # 替换为你的txt文件路径
#
# # 创建一个空的无向图
# G = nx.Graph()
#
# # 读取txt文件中的数据
# with open(file_path, "r", encoding="utf-16") as file:
#     lines = file.readlines()
#
# # with open(file_path, "rb") as file:
# #     raw_data = file.read()
# #     result = chardet.detect(raw_data)
# #     encoding = result['encoding']
# #     print(f"检测到的文件编码是：{encoding}")
# #
# # # # 添加边到图中
# for line in lines:
#     node1, node2 = map(int, line.split())
#     G.add_edge(node1, node2)
#
# print(G.number_of_nodes())
# # 将图存储到二进制文件中
# with open("2_following", "wb") as f:
#     pickle.dump(G, f)
#
# print("无向图已成功生成并存储在 'network_graph_undirected.pkl' 文件中。")




with open("douban", "rb") as f:
    G = pickle.load(f)

# 找出所有的孤立点
isolated_nodes = list(nx.isolates(G))

# 打印结果
if isolated_nodes:
    print(f"图中存在 {len(isolated_nodes)} 个孤立点：")
    print(isolated_nodes)
else:
    print("图中没有孤立点。")




# # 读取数据并添加边
# file_path = r"C:\Users\11792\Desktop\arXiv\groundtruth.txt"  # 替换为你的txt文件路径
# output_file_path = r"C:\Users\11792\Desktop\arXiv\anchor.txt"  # 输出文件路径
#
# # 创建一个空的列表来存储边
# edges = []
#
# # 读取txt文件中的数据
# with open(file_path, "r", encoding="utf-16") as file:
#     lines = file.readlines()
#
# # 将数据存储到列表中
# for line in lines:
#     node1, node2 = map(int, line.split())
#     edges.append([node1, node2])
#
# # 将列表写入新的txt文件
# with open(output_file_path, "w") as file:
#     file.write(str(edges))
#
# print(f"数据已成功写入 '{output_file_path}' 文件中。")
