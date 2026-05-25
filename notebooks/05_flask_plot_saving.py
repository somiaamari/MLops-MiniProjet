from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)

# Load and process data (same as your current code)
file_path = "../data/smart_home_dataset.csv"
data = pd.read_csv(file_path)
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
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Route: Homepage with visualizations and top rules
@app.route('/')
def home():
    # Prepare visualizations
    # Generate bar chart for top rules' confidence
    top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_rules['antecedents'].astype(str), top_rules['confidence'], color='skyblue')
    ax.set_xlabel('Confidence')
    ax.set_title('Top 10 Association Rules by Confidence')
    plt.tight_layout()

    # Save the plot as a PNG file to display on the webpage
    plot_path = 'static/top_rules_confidence.png'
    fig.savefig(plot_path)

    return render_template('index.html', top_rules=top_rules.to_dict('records'), plot_path=plot_path)

# Route: Predict recommendations based on user input
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json['items']  # List of items from the user
    matching_rules = rules[rules['antecedents'].apply(lambda x: set(user_input).issubset(x))]
    recommendations = matching_rules.sort_values(by='confidence', ascending=False)['consequents'].head(5).tolist()
    recommendations = [list(rec) for rec in recommendations]  # Convert frozenset to list
    return jsonify(recommendations=recommendations)

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
    app.run(debug=True)
