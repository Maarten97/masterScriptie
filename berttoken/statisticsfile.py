# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'your_data.csv' with the actual path to your CSV file
# Assuming the CSV has a column 'sentence_length'
df = pd.read_csv('loglarge2.csv', usecols=['sentence_length'])

# Filter out rows where sentence_length is 0
df = df[df['sentence_length'] != 0]

# Compute Descriptive Statistics
mean_length = df['sentence_length'].mean()
median_length = df['sentence_length'].median()
mode_length = df['sentence_length'].mode()[0]  # Mode can have multiple values, take the first
std_dev_length = df['sentence_length'].std()
min_length = df['sentence_length'].min()
max_length = df['sentence_length'].max()
quantiles = df['sentence_length'].quantile([0.25, 0.50, 0.75])

# Print the statistics
print("Descriptive Statistics (Excluding sentence_length = 0):")
print(f"Mean: {mean_length}")
print(f"Median: {median_length}")
print(f"Mode: {mode_length}")
print(f"Standard Deviation: {std_dev_length}")
print(f"Minimum: {min_length}")
print(f"Maximum: {max_length}")
print(f"25th Percentile: {quantiles[0.25]}")
print(f"50th Percentile (Median): {quantiles[0.50]}")
print(f"75th Percentile: {quantiles[0.75]}")

# Plotting the Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['sentence_length'], bins=100, color='blue', alpha=0.7)
plt.title('Sentence Length Distribution (Excluding sentence_length = 0)')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plotting the Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['sentence_length'], color='green')
plt.title('Boxplot of Sentence Length (Excluding sentence_length = 0)')
plt.xlabel('Sentence Length')
plt.grid(True)
plt.show()

# Import necessary libraries
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# # Replace 'your_data.csv' with the actual path to your CSV file
# df = pd.read_csv('log2.csv', usecols=['Word Count'])
#
# # Filter out rows where sentence_length is 0
# df = df[df['Word Count'] != 0]
#
# # Create bins for sentence lengths with a step of 5 and an 'above 100' category
# bins = list(range(0, 101, 5)) + [float('inf')]  # Bins: 0-5, 6-10, ..., 96-100, 100+
# labels = [f"{i}-{i+4}" for i in range(0, 100, 5)] + ['100+']
#
# # Bin the sentence lengths
# df['length_category'] = pd.cut(df['Word Count'], bins=bins, labels=labels, right=False)
#
# # Compute frequency of each bin
# bin_counts = df['Word Count'].value_counts(sort=True)
#
# # Print the bin counts
# print("\nFrequency of Sentence Length Categories:")
# print(bin_counts)
#
# # Plotting the Histogram (Categorical Frequency)
# plt.figure(figsize=(10, 6))
# bin_counts.plot(kind='bar', color='blue', alpha=0.7)
# plt.title('Sentence Length Distribution in 5-word Bins (Excluding sentence_length = 0)')
# plt.xlabel('Sentence Length (in Bins)')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plotting the Boxplot (for the unbinned data)
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=df['Word Count'], color='green')
# plt.title('Boxplot of Sentence Length (Excluding sentence_length = 0)')
# plt.xlabel('Sentence Length')
# plt.grid(True)
# plt.show()
