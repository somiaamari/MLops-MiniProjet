from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# Create Flask app
app = Flask(__name__)

# Ensure the static directory exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

# Load and optimize the dataset
file_path = "../data/smart_home_dataset.csv"
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

# Prepare data for association rules
data_for_apriori = data.drop(columns=["Activity", "timestamp"])
transactions = data_for_apriori.apply(
    lambda row: [col for col, value in row.items() if value == 1], axis=1
).tolist()

# Encode transactions for Apriori algorithm
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets and rules
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Route: Homepage
@app.route('/')
def home():
    global rules
    top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
    
    # Generate a plot for top rules
    plot_path = 'static/top_rules_confidence.png'
    if not os.path.exists(plot_path):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_rules['antecedents'].astype(str), top_rules['confidence'], color='skyblue')
        ax.set_xlabel('Confidence')
        ax.set_title('Top 10 Association Rules by Confidence')
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    return render_template('index.html', top_rules=top_rules.to_dict('records'), plot_path=plot_path)

# Route: Generate recommendations based on user input
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if not request.json or 'user_data' not in request.json:
            return jsonify({"error": "'user_data' key missing from request"}), 400
        
        user_data = request.json['user_data']
        activity = user_data.get('activity', 'sleep')  # Default to 'sleep'
        timestamp = user_data.get('timestamp')
        user_input = [key for key, value in user_data.items() if value == '1']

        # Filter data by activity
        filtered_data = data[data['Activity'] == activity]
        transactions = filtered_data.drop(columns=["Activity", "timestamp"]).apply(
            lambda row: [col for col, value in row.items() if value == 1], axis=1
        ).tolist()

        # Recreate rules for this activity
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
        activity_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

        # Match user input with rules
        matching_rules = activity_rules[activity_rules['antecedents'].apply(lambda x: set(user_input).issubset(x))]
        recommendations = matching_rules.sort_values(by='confidence', ascending=False)['consequents'].head(5).tolist()
        recommendations = [list(rec) for rec in recommendations]  # Convert frozensets to lists

        return jsonify(recommendations=recommendations, activity=activity, timestamp=timestamp)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Add new data
@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        new_data = {
            'timestamp': request.form['timestamp'],
            'Activity': request.form['activity'],
            'mainDoorLock': int(request.form['mainDoorLock']),
            'bed': int(request.form['bed']),
            'bedroomCarp': int(request.form['bedroomCarp']),
            'wardrobe': int(request.form['wardrobe']),
            'tv': int(request.form['tv']),
            'oven': int(request.form['oven']),
            'officeLight': int(request.form['officeLight']),
        }

        global data
        data = data.append(new_data, ignore_index=True)
        data.to_csv(file_path, index=False)

        return redirect(url_for('home'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
