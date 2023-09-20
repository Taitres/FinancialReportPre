import os

import pdfplumber
import re
import jieba

# 打开PDF文件
# 指定PDF文件所在的目录
pdf_directory = './train/'
output_directory = './train_data/'  # 输出文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_directory, exist_ok=True)
# 遍历指定目录下的PDF文件
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_filepath = os.path.join(pdf_directory, filename)
        output_filepath = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')  # 生成对应的.txt文件名

        # 提取PDF文件内容并清洗
        with pdfplumber.open(pdf_filepath) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

    # 使用jieba进行分词
    seg_list = jieba.cut(text, cut_all=False)
    cleaned_text = " ".join(seg_list)

    # 去除特殊字符和标点符号
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', cleaned_text)

    # 转换文本为小写
    cleaned_text = cleaned_text.lower()

    stop_words = ["的", "了", "是", "我", "你",'在']  # 自定义停用词列表
    words = cleaned_text.split()
    filtered_text = [word for word in words if word not in stop_words]

    cleaned_text = ' '.join(filtered_text)

    # 将清洗后的文本保存为.txt文件
    with open(output_filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(cleaned_text)

    print(f'已处理文件: {pdf_filepath}，并保存为: {output_filepath}')