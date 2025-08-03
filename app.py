import os
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

# Data
data = [
    {"headline": "NASA launches new telescope into orbit", "keywords": ["nasa", "telescope", "orbit"]},
    {"headline": "Stock market crashes after inflation report", "keywords": ["stock", "market", "inflation"]},
    {"headline": "Google unveils new AI tools for developers", "keywords": ["google", "ai", "developers"]},
    {"headline": "Climate change summit discusses new strategies", "keywords": ["climate", "change", "summit"]},
]

corpus = [item["headline"].lower() for item in data]
all_keywords = [item["keywords"] for item in data]

# BERT Embedding
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
X = bert_model.encode(corpus)

# Labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(all_keywords)

# Train Model
model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X, Y)

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        headline = request.form["headline"]
        X_test = bert_model.encode([headline.lower()])
        Y_proba = model.predict_proba(X_test)
        probs = np.array([p[0][1] for p in Y_proba])
        word_probs = dict(zip(mlb.classes_, probs))
        top_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Plot
        words, scores = zip(*top_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, scores, color='skyblue')
        plt.title("Top Predicted Keywords with Confidence Scores")
        plt.ylabel("Probability")
        plt.xlabel("Word")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = "static/plot.png"
        plt.savefig(plot_path)
        plt.close()

        return render_template("index.html", headline=headline, top_words=top_words, plot_url=plot_path)

    return render_template("index.html", headline="", top_words=[], plot_url="")

if __name__ == "__main__":
    app.run(debug=True)
