from datasets import load_dataset
from collections import defaultdict

# 配置
DATASET_NAME = "Attention1115/DLR-Web"  # 如 xxx/yyy
SPLIT = "full"                  # 划分：train/validation/test
CLASS_FIELD = "discipline"    # 75分类的字段
SAMPLE_PER_CLASS = 200           # 每类200条
save_path = "sampled_200perclass"

# 计数器：key=分类值，value=已采样条数
class_counter = defaultdict(int)
sample_list = []

# 关键 streaming=True：不下载40G全量，边流式读取边筛选
ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

for item in ds:
    cls = item[CLASS_FIELD]
    if class_counter[cls] < SAMPLE_PER_CLASS:
        sample_list.append(item)
        class_counter[cls] += 1
        # 全部采满提前退出（70*200=14000）
        if len(sample_list) >= 75 * SAMPLE_PER_CLASS:
            break

# 转为Dataset对象并保存（可选：存本地json/parquet）
from datasets import Dataset
ds_sample = Dataset.from_list(sample_list)
# 保存数据集
ds_sample.save_to_disk(save_path)
# 导出json
ds_sample.to_json(f"{save_path}.json")

print(f"总共采样：{len(sample_list)}")
print("各类采样数量：", dict(class_counter))