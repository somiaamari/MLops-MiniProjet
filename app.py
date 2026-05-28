from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

if not os.path.exists('static'):
    os.makedirs('static')


app = Flask(__name__)

# Ensure the static directory exists for saving the plot
if not os.path.exists('static'):
    os.makedirs('static')

# Load and process data (same as your current code)
file_path = "data/smart_home_dataset.csv"
data = pd.read_csv(file_path)
def optimize_data(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

data = optimize_data(data)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H_%M_%S')

data.fillna(0, inplace=True)
data_for_apriori = data.drop(columns=["Activity", "timestamp"])
transactions = data_for_apriori.apply(
    lambda row: [col for col, value in row.items() if value == 1], axis=1
).tolist()

# Convert to encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets and rules
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
num_itemsets = len(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=num_itemsets)

# Route: Homepage with visualizations and top rules
@app.route('/')
def home():
    global rules
    top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
    
    # Check if plot exists; only create if it doesn't
    plot_path = 'static/top_rules_confidence.png'
    if not os.path.exists(plot_path):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_rules['antecedents'].astype(str), top_rules['confidence'], color='skyblue')
        ax.set_xlabel('Confidence')
        ax.set_title('Top 10 Association Rules by Confidence')
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)  # Free memory by closing the plot

    return render_template('index.html', top_rules=top_rules.to_dict('records'), plot_path=plot_path)

# Route: Predict recommendations based on newly submitted data
# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         # Log the raw request data to check what the frontend is sending
#         print("Raw data:", request.data)  # This will print the raw body of the request
#         print("Parsed JSON:", request.json)  # This will print the parsed JSON data

#         # Check if request.json is None or if it does not have 'items' key
#         if not request.json or 'items' not in request.json:
#             return jsonify({"error": "'items' key missing from request"}), 400
        
#         # Extract 'items' from the JSON body
#         user_input = request.json['items']  # List of items from the user
#         print("User Input:", user_input)

#         # Proceed with recommendation logic
#         matching_rules = rules[rules['antecedents'].apply(lambda x: set(user_input).issubset(x))]
#         recommendations = matching_rules.sort_values(by='confidence', ascending=False)['consequents'].head(5).tolist()
#         recommendations = [list(rec) for rec in recommendations]  # Convert frozenset to list

#         return jsonify(recommendations=recommendations)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        print("Raw data:", request.data)
        print("Parsed JSON:", request.json)

        # Extract user_data
        if not request.json or 'user_data' not in request.json:
            return jsonify({"error": "'user_data' key missing from request"}), 400
        
        user_data = request.json['user_data']

        # Convert the user_data dictionary into a list of active items
        user_input = [key for key, value in user_data.items() if value == '1']  # Ensure values are strings '1'
        print("User Input (active items):", user_input)

        # Proceed with recommendation logic
        matching_rules = rules[rules['antecedents'].apply(lambda x: set(user_input).issubset(x))]
        recommendations = matching_rules.sort_values(by='confidence', ascending=False)['consequents'].head(5).tolist()
        recommendations = [list(rec) for rec in recommendations]
        
        print("recommendations:", recommendations)

        return jsonify(recommendations=recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Add new data
@app.route('/add_data', methods=['POST'])
def add_data():
    if request.method == 'POST':
        timestamp = request.form['timestamp']
        activity = request.form['activity']
        mainDoorLock = int(request.form['mainDoorLock'])
        bed = int(request.form['bed'])
        bedroomCarp = int(request.form['bedroomCarp'])

        new_data = {
            'timestamp': timestamp,
            'activity': activity,
            'mainDoorLock': mainDoorLock,
            'bed': bed,
            'bedroomCarp': bedroomCarp,
        }

        global data
        data = data.append(new_data, ignore_index=True)
        data.to_csv(file_path, index=False)

        return redirect(url_for('home'))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

