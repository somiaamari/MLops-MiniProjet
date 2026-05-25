from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)

# Load dataset and preprocess
file_path = "../data/smart_home_dataset.csv"
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H_%M_%S')
data.fillna(0, inplace=True)
data_for_apriori = data.drop(columns=["Activity", "timestamp"])

# Convert to transactional format
transactions = data_for_apriori.apply(
    lambda row: [col for col, value in row.items() if value == 1], axis=1
).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori and rules generation
min_support = 0.1
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
min_confidence = 0.7
num_itemsets = len(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)

# Format rules for readability
formatted_rules = [
    {
        "rule": f"If {', '.join(list(row['antecedents']))} then {', '.join(list(row['consequents']))}",
        "support": round(row["support"] * 100, 2),
        "confidence": round(row["confidence"] * 100, 2),
        "lift": round(row["lift"], 2),
    }
    for _, row in rules.iterrows()
]

@app.route("/")
def home():
    return render_template("index.html", frequent_itemsets=frequent_itemsets.head(10).to_dict("records"))

@app.route("/rules")
def show_rules():
    return render_template("rules.html", rules=formatted_rules)

if __name__ == "__main__":
    app.run(debug=True)
