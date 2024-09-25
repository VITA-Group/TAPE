import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import scienceplots
# plt.style.use(['notebook'])
plt.style.use(['notebook'])
# import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 

SMALL_SIZE = 28
MEDIUM_SIZE = 30
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# 读取CSV文件
df = pd.read_csv('results.csv')

# 获取唯一的data和model值
data_values = df['data'].unique()
model_values = df['model'].unique()
# model_values = model_values[model_values != 'adape']

# 设置子图
fig, axes = plt.subplots(nrows=1, ncols=len(data_values), figsize=(8 * len(data_values), 6))

# 如果只有一个子图，axes不再是数组，需要转成数组
if len(data_values) == 1:
    axes = [axes]

legend_mapping = {
    'alibi': 'ALiBi',
    'rope': 'RoPE',
    'adape': 'AdeRoPE',
    # 添加其他模型的映射
}
data_mapping = {
    "pg19": "PG19",
    "arxiv": "ArXiv",
    "github": "Github"
}

# 为每个data值创建一个子图
for i, data in enumerate(data_values):
    ax = axes[i]
    for model in model_values:
        subset = df[(df['data'] == data) & (df['model'] == model)]
        log_perplexity = np.log(subset['perplexity'])
        ax.plot(subset['block_size'], log_perplexity, label=legend_mapping.get(model, model), marker='o', markersize=12, linewidth=3)
        # ax.plot(subset['block_size'], subset['perplexity'], label=model)
    
    ax.set_title(f'{data_mapping.get(data, data)}')
    # ax.set_yscale('log')
    ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])
    if i == 1:
        ax.set_xlabel(r'Validation sequence length ($\times 10^3$)')  # 改变横坐标名称
    if i == 0:
        ax.legend()
        ax.set_ylabel('Log Perplexity')
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 三连
plt.tight_layout()
plt.savefig('pretrain_ppl.pdf', dpi=300, transparent=True)
plt.show()