import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load dataset
file_path = "../data/smart_home_dataset.csv"
data = pd.read_csv(file_path)

# Convert timestamp to datetime for further analysis (if needed)
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H_%M_%S')

# Check for null values and fill if necessary
data.fillna(0, inplace=True)

# Drop irrelevant columns
data_for_apriori = data.drop(columns=["Activity", "timestamp"])

# Convert data to transactional format
transactions = data_for_apriori.apply(
    lambda row: [col for col, value in row.items() if value == 1], axis=1
).tolist()

# Encode transactions for Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.1  # Adjust based on dataset size and domain
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Display top frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets.sort_values(by="support", ascending=False).head(10))
num_itemsets = len(frequent_itemsets)

# Generate association rules
min_confidence = 0.7  # Adjust based on application needs
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence,  num_itemsets=num_itemsets)

# Display top rules
print("Top Association Rules:")
print(rules.sort_values(by="confidence", ascending=False).head(10))

# Visualize top frequent itemsets by support
top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(x))
top_itemsets.plot(x='itemsets', y='support', kind='bar', figsize=(12, 6), legend=False)
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize top rules by confidence
top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.bar(
    x=range(len(top_rules)),
    height=top_rules['confidence'],
    tick_label=[f"{', '.join(list(ant))} → {', '.join(list(con))}"
                for ant, con in zip(top_rules['antecedents'], top_rules['consequents'])]
)
plt.title('Top 10 Association Rules by Confidence')
plt.xlabel('Rules')
plt.ylabel('Confidence')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

filtered_rules = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.7)]
print(filtered_rules)

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for _, row in filtered_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent)

nx.draw(G, with_labels=True, node_size=3000, node_color='lightblue')
plt.show()
transactions = []

for _, row in data.iterrows():
    transaction = row[row == 1].index.tolist()  # Get column names where value is 1
    transactions.append(transaction)

# Convert transactions into a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

# Step 3: Apply FP-Growth
min_support = 0.2  # Adjust as needed
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate association rules
min_confidence = 0.6  # Adjust as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)

print("\nAssociation Rules:")
print(rules)

from flask import jsonify




@app.route('/get_rules')
def get_rules():
    # Convert association rules to a JSON-serializable format
    serializable_rules = []
    for _, row in rules.iterrows():
        serializable_rules.append({
            "antecedents": list(row["antecedents"]),  # Convert frozenset to list
            "consequents": list(row["consequents"]),  # Convert frozenset to list
            "support": row["support"],
            "confidence": row["confidence"],
            "lift": row["lift"],
            "leverage": row["leverage"],
            "conviction": row["conviction"],
        })

    return jsonify(serializable_rules)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
