import pickle
import random

def select_captions_from_pkl(pkl_path, txt_path, num_captions=2000):
    """
    从pkl文件中读取字典，随机选取指定数量的caption，保存到txt文件（每行一个caption）
    自动处理值为列表的情况（将列表元素拼接为字符串）
    """
    # 读取pkl文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 确保data是字典
    if not isinstance(data, dict):
        raise TypeError("pkl文件中的对象必须是字典")

    # 处理每个value，统一转换为字符串
    captions = []
    for key, value in data.items():
        if isinstance(value, str):
            caption = value
        elif isinstance(value, list):
            # 如果值是列表，将元素用空格连接（假设元素是字符串或可转换为字符串）
            caption = ' '.join(str(item) for item in value)
        else:
            # 其他类型直接转为字符串
            caption = str(value)
        captions.append(caption)

    # 检查数量
    if len(captions) < num_captions:
        print(f"警告：字典中只有{len(captions)}个物体，少于所需的{num_captions}个，将全部输出。")
        num_captions = len(captions)

    # 随机选择指定数量的caption
    selected_captions = random.sample(captions, num_captions)

    # 写入txt文件，每行一个caption，并清理内部的换行符
    with open(txt_path, 'w', encoding='utf-8') as f:
        for caption in selected_captions:
            # 将caption内部的换行符替换为空格，确保每行是一个完整的文本块
            caption_clean = caption.replace('\n', ' ').replace('\r', ' ')
            f.write(caption_clean + '\n')

    print(f"成功将{num_captions}个caption写入文件：{txt_path}")

# 使用示例
if __name__ == "__main__":
    pkl_file = "./Cap3D_human_Objaverse.pkl"   # 请替换为您的pkl文件路径
    output_txt = "./example_captions.txt"  # 输出文件名
    select_captions_from_pkl(pkl_file, output_txt, 5000)