import matplotlib.pyplot as plt

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Provided data for three methods
longlora = {'890': 0.5, '1982': 0.3, '2801': 0.3, '3894': 0.3, '4986': 0.3, '5805': 0.3, '6898': 0.3, '7990': 0.4}
lora = {'890': 0.5, '1982': 0.3, '2801': 0.3, '3894': 0.3, '4986': 0.3, '5805': 0.3, '6898': 0.3, '7990': 0.4}
adape = {'890': 1.0, '1982': 1.0, '2801': 1.0, '3894': 1.0, '4986': 1.0, '5805': 0.9, '6898': 1.0, '7990': 1.0}

# X-axis values (1k to 8k)
x_labels = ['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k']

# Extracting values for y-axis from dictionaries
longlora_values = list(longlora.values())
lora_values = list(lora.values())
adape_values = list(adape.values())

# Plotting
plt.figure(figsize=(8, 5))

# Plot each method's data
plt.plot(x_labels, longlora_values, label="LongLoRA", marker='o', linestyle='-', color='blue')
plt.plot(x_labels, lora_values, label="LoRA", marker='o', linestyle='--', color='green')
plt.plot(x_labels, adape_values, label="Adape", marker='o', linestyle='-', color='orange')

# Adding titles and labels
plt.title("Passkey Retrieval Accuracy")
# plt.xlabel("Context Length")
# plt.ylabel("Accuracy")

# Adding grid only for horizontal lines (no vertical lines)
# plt.xticks(range(len(x_labels)))
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.grid(axis='y', linestyle='--')
# plt.grid(axis='y')

# Adding legend
# plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('retrieval.png', dpi=300, transparent=True)
plt.show()