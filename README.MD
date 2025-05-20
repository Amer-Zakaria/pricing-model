# Mobile Pricing Decision Boundary Project

## Overview
[Plot Preview](https://smartphone-pricing.streamlit.app) 

This project uses a Gaussian Naive Bayes classifier to help mobile manufacturers price new phones based on performance(CPU, and CPU/RAM interaction) and the current market conditions. The model calculates a "score per dollar" ratio for each smartphone and classifies them as underpriced or overpriced. An interactive visualization shows a decision boundary, with color intensity representing the classifier's confidence:

- **Deeper green:** Higher confidence that a phone is underpriced.
- **Deeper red:** Higher confidence that a phone is overpriced.

## Data & Methodology

- **Data:** A sample set of smartphone specifications brought from scraping GSM Arena, smartphones are from 2024 to May 2025. The most feature that effect the price has been extracted, which turned out to the performance score from Aututu.
- **Model:** A Gaussian Naive Bayes classifier is trained on the data to determine the decision boundary.
- **Visualization:** The decision boundary is plotted, with color intensity reflecting the probability (confidence) of being underpriced or overpriced.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
      git clone https://github.com/Amer-Zakaria/pricing-model.git
      cd pricing-model
   ```

2. **Install necessary dependencies:**

   ```bash
      pip install -r requirements.txt
   ```

3. **Run the code:**
   Run directly from the Jupyter Notebook file
