import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_processed.csv')
print(df.shape)
label_value_counts = df.label.value_counts()

plt.figure(figsize=(35, 10))
plt.bar(label_value_counts.index, label_value_counts.values, color='skyblue', width=0.5)

plt.yscale('log')
plt.tight_layout()
plt.xlabel('Label')
plt.ylabel('Frequency (log scale)')
plt.title('Histogram of label distribution')
# TODO average word per radio message as well as an example of a transripted message to put in the paper
plt.savefig('label_histogram.png')
plt.show()
