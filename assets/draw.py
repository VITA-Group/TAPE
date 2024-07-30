import pandas as pd
import matplotlib.pyplot as plt

SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 32

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

# 设置子图
fig, axes = plt.subplots(nrows=len(data_values), ncols=1, figsize=(10, 5 * len(data_values)))

# 如果只有一个子图，axes不再是数组，需要转成数组
if len(data_values) == 1:
    axes = [axes]

# 为每个data值创建一个子图
for i, data in enumerate(data_values):
    ax = axes[i]
    for model in model_values:
        subset = df[(df['data'] == data) & (df['model'] == model)]
        ax.plot(subset['block_size'], subset['perplexity'], marker='o', label=model)
    
    ax.set_title(f'Data: {data}')
    ax.set_xlabel('Sequence Length')  # 改变横坐标名称
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('results_plot.png', dpi=300, transparent=True)

# 显示图形
plt.show()