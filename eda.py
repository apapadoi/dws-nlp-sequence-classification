import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

df = pd.read_csv('data_processed.csv')
print(df.shape)
label_value_counts = df.label.value_counts()

plt.figure(figsize=(28, 5))
plt.bar(label_value_counts.index, label_value_counts.values, width=0.5)

plt.yscale('log')
plt.xlabel('Label')
plt.ylabel('Frequency (log scale)')
plt.title('Histogram of label distribution')
# TODO average word count per radio message as well as an example of a transripted message to put in the paper
plt.savefig('label_histogram.png')
plt.show()
