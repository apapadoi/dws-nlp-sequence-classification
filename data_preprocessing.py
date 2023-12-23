import pandas as pd

df = pd.read_csv('data.csv')

df_without_na = df.dropna()
print(f'Num of rows containing NaN: {df.shape[0] - df_without_na.shape[0]}')
print(f'Df without NaN shape: {df_without_na.shape}')

duplicates = df_without_na[df_without_na.duplicated(subset=['text', 'label'])]
print(f'Total duplicate rows: {duplicates.shape[0]}')

df = df_without_na.drop_duplicates(subset=['text', 'label'])

print(f'Processed df shape: {df.shape}')

label_counts = df.label.value_counts()
less_than_10_occurrences = label_counts[label_counts < 10].index.tolist()

filtered_df = df[~df.label.isin(less_than_10_occurrences)]

splitted_features = filtered_df.features.str.split(',', expand=True).astype(float)

filtered_df.loc[:, 'floats_per_row_count'] = filtered_df.features.str.split(',').apply(lambda x: len(x))

statistics = filtered_df['floats_per_row_count'].describe()
print("\nStatistics about the number of floats per row:")
print(statistics)

splitted_features = splitted_features.apply(pd.to_numeric, errors='coerce')
max_features = splitted_features.shape[1]

num_stft = 1025
num_spectrogram = 128
num_mfccs = 13
num_pitches = 1025
num_energy = 1

new_columns = ([f'stft_{i+1}' for i in range(num_stft)] +
               [f'spectrogram_{i+1}' for i in range(num_spectrogram)] +
               [f'mfccs_{i+1}' for i in range(num_mfccs)] +
               [f'pitches_{i+1}' for i in range(num_pitches)] +
               [f'energy_{i+1}' for i in range(num_energy)])

splitted_features.columns = new_columns
filtered_df = pd.concat([filtered_df, splitted_features], axis=1)

filtered_df.drop(columns=['features', 'floats_per_row_count'], inplace=True)
filtered_df['text'] = filtered_df['text'].str.lower()

filtered_df.to_csv('data_processed.csv', index=False)
