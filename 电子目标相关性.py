import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 任务关联数据
data = {
    '目标类型': ['雷达站A', '雷达站B', '飞机C', '飞机D', '导弹装置X', '导弹装置Y'],
    '雷达关联': [6, 4, 9, 2, 8, 5],
    '导弹关联': [8, 5, 2, 6, 7, 4],
    '飞机关联': [3, 7, 6, 4, 5, 2]
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 将数据进行melt操作，使之适合绘制热力图
df_melted = df.melt(id_vars='目标类型', var_name='关联类型', value_name='关联强度')

# 使用seaborn绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(data=df_melted.pivot('目标类型', '关联类型', '关联强度'), cmap='Blues', annot=True, fmt='.1f', linewidths=0.5)
plt.title('军事电子目标关联关系热力图')
plt.xlabel('关联类型')
plt.ylabel('目标类型')
plt.show()
