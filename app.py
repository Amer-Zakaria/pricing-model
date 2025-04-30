# %%
import json
import numpy as np
import streamlit as st
import random

st.set_page_config(page_title="Mobile Pricing Decision Boundary", layout="wide")

# %% [markdown]
# ## Prepairing a Sample
#

# %%
with open("sample.json", "r") as file:
    sample = json.load(file)
sample_original = sample.copy()
random.seed(42)
random.shuffle(sample)

# Create feature and target arrays
X = np.array([[item["score"], item["price"]] for item in sample])
avg_spd = sum(item["score_per_dollar"] for item in sample) / len(sample)
y = [1 if item["score_per_dollar"] > avg_spd else 0 for item in sample]

# %% [markdown]
# ## Training the Classifier
#

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", GaussianNB(var_smoothing=1e-9)),
    ]
)
mod = GridSearchCV(
    estimator=pipe,
    param_grid={},
    cv=3,
)
mod.fit(X, y)
pd.DataFrame(mod.cv_results_).to_csv("cv_results.csv", index=False)

# %% [markdown]
# ## Visualizing Decision Boundaries with Confidence Intesity
#

# %%
import plotly.graph_objects as go
import streamlit as st
from IPython.display import HTML

# 1. Define the mesh grid
x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
y_min, y_max = X[:, 1].min() - 25, X[:, 1].max() + 25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 5))

# 2. Predict probabilities over the grid
probs = mod.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
probs = probs.reshape(xx.shape)

# 3. Create heatmap (probability background)
red = "#E74C3C"  # a softer tomato-red
green = "#2ECC71"  # a fresh mint-green
heatmap = go.Contour(
    x=np.arange(x_min, x_max, 10),
    y=np.arange(y_min, y_max, 5),
    z=probs,
    #  ["#FF0000", "#00AA00"]
    colorscale=[(0, red), (1, green)],
    opacity=0.8,
    colorbar=dict(
        title="Underpriced Probability",
        x=1.1,
    ),
)

# 4. Add scatter plot (actual data points)
# extract a parallel list of names
smartphoneNames = [s["name"] for s in sample_original]
scatter = go.Scatter(
    x=X[:, 0],
    y=X[:, 1],
    mode="markers",
    marker=dict(
        size=8,
        color=y,
        colorscale=[red, green],
        line=dict(width=1, color="black"),
    ),
    name="Smartphones",
    text=[f"Label: {label}" for label in y],
    customdata=smartphoneNames,
    hovertemplate=(
        "Name: %{customdata}<br>" "Performance: %{x}<br>" "Price: $%{y}<extra></extra>"
    ),
)

# 5. Combine everything into one figure
fig = go.Figure(data=[heatmap, scatter])
fig.update_layout(
    title="Price Intensity Based on Mobile Performance",
    xaxis_title="Performance Score",
    yaxis_title="Starting Price in USD",
    width=800,
    height=600,
)

# 6. Show in Streamlit
st.plotly_chart(fig, use_container_width=True)
# html = fig.to_html(include_plotlyjs="cdn")
# HTML(html)
