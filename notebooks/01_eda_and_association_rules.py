import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

print(association_rules.__module__)
# Load dataset
file_path = "../data/smart_home_dataset.csv"
data = pd.read_csv(file_path)

# Inspect the first few rows
print(data.head())

# Check dataset summary
print(data.info())

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H_%M_%S')


print(data.isnull().sum())

# If there are missing values, you can either drop or fill them
# Example: Filling with 0 (assuming binary columns)
data.fillna(0, inplace=True)

# Encode 'Activity'
label_encoder = LabelEncoder()
data['Activity'] = label_encoder.fit_transform(data['Activity'])

# Optional: View the mapping
activity_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(activity_mapping)

# Exclude 'Activity' and 'timestamp' for normalization
columns_to_normalize = data.drop(['Activity', 'timestamp'], axis=1).columns
scaler = MinMaxScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Plot activity counts
plt.figure(figsize=(10, 5))
sns.countplot(x='Activity', data=data)
plt.title('Activity Frequency')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(ticks=range(len(activity_mapping)), labels=label_encoder.inverse_transform(range(len(activity_mapping))))
plt.show()

# Calculate correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Example: Plot `livingLight` state over time
plt.figure(figsize=(15, 5))
plt.plot(data['timestamp'], data['livingLight'], label='Living Light')
plt.title('Living Light State Over Time')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Apply PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.drop(['Activity', 'timestamp'], axis=1))

# Visualize reduced data
plt.figure(figsize=(10, 7))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data['Activity'], palette='Set2', s=50)
plt.title('PCA Visualization of Activities')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Activity', labels=label_encoder.inverse_transform(range(len(activity_mapping))))
plt.show()


# from pycaret.arules import *

# s = setup(data = data, transaction_id = 'InvoiceNo', item_id = 'Description')

# arules = create_model(metric='confidence', threshold=0.5, min_support=0.05)


# Remove irrelevant columns
devices_data = data.drop(columns=["Activity", "timestamp"])

# Create transactions: each row is converted to a list of active (ON) devices
transactions = devices_data.apply(
    lambda row: [devices_data.columns[i] for i in range(len(row)) if row[i] == 1],
    axis=1
).tolist()

# Encode transactions as one-hot
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


# Find frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
print(frequent_itemsets.head())
# Display frequent itemsets
print(frequent_itemsets)
num_itemsets = len(frequent_itemsets)


# Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=num_itemsets)

# Display rules
print(rules)

frequent_itemsets.sort_values(by='support', ascending=False).head(10).plot(
    x='itemsets', y='support', kind='bar', figsize=(10, 5)
)
plt.show()

# Convert itemsets to strings for better readability
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(x))

# Sort by support and take the top 10
top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)

# Plot
top_itemsets.plot(
    x='itemsets', 
    y='support', 
    kind='bar', 
    figsize=(12, 6), 
    legend=False
)

# Add labels and title
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()

# Sort rules by confidence and take the top 10
top_rules = rules.sort_values(by='confidence', ascending=False).head(10)

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.bar(
    x=range(len(top_rules)), 
    height=top_rules['confidence'], 
    tick_label=[', '.join(list(ant)) + ' → ' + ', '.join(list(con)) 
                for ant, con in zip(top_rules['antecedents'], top_rules['consequents'])]
)

# Add labels and title
plt.title('Top 10 Association Rules by Confidence')
plt.xlabel('Rules')
plt.ylabel('Confidence')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()
plt.show()

# Convert itemsets to strings
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(x))

# Plot top 10 frequent itemsets by support
frequent_itemsets.sort_values(by='support', ascending=False).head(10).plot(
    x='itemsets', 
    y='support', 
    kind='bar', 
    figsize=(12, 6),
    legend=False
)
plt.title('Top 10 Frequent Itemsets')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
plt.tight_layout()
plt.show()