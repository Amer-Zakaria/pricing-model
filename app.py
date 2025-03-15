import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Mobile Pricing Decision Boundary", layout="wide")

# ========================
# PREPARING A SAMPLE
# ========================

with open("sample.json", "r") as file:
    sample = json.load(file)

# Create feature and target arrays
X = np.array([[item["score"], item["price"]] for item in sample])
avg_spd = sum(item["score_per_dollar"] for item in sample) / len(sample)
y = [1 if item["score_per_dollar"] > avg_spd else 0 for item in sample]

# ========================
# TRAINING THE CLASSIFIER
# ========================
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, y)

# ========================
# VISUALIZING DECISION BOUNDARIES WITH CONFIDENCE INTENSITY
# ========================

# Define the grid for plotting
x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
y_min, y_max = X[:, 1].min() - 25, X[:, 1].max() + 25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 5))

# Get the predicted probabilities for the "underpriced" class (class 1)
probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
probs = probs.reshape(xx.shape)

# Create a custom colormap from red (overpriced) to green (underpriced)
cmap_custom = LinearSegmentedColormap.from_list("RedGreen", ["#FF0000", "#00AA00"])

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, probs, alpha=0.4, cmap=cmap_custom)

# Plot the training data with colors reflecting their class confidence
cmap_bold = LinearSegmentedColormap.from_list("RedGreen", ["#FF0000", "#00AA00"])
ax.scatter(
    X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=50, label="Smartphone"
)

ax.set_xlabel("Performance Score")
ax.set_ylabel("Starting Price in USD")
ax.set_title("Price intensity based on mobile performance")
ax.legend()

st.pyplot(fig)
