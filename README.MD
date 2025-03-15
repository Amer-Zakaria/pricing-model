# Mobile Pricing Decision Boundary Project

## Overview

This project uses a Gaussian Naive Bayes classifier to help mobile manufacturers price new phones based on performance(CPU, and CPU/RAM interaction) and the current market conditions. The model calculates a "score per dollar" ratio for each smartphone and classifies them as underpriced or overpriced. An interactive visualization shows a decision boundary, with color intensity representing the classifier's confidence:

- **Deep green:** High confidence that a phone is underpriced.
- **Deep red:** High confidence that a phone is overpriced.

## Data & Methodology

- **Data:** A sample set of smartphone specifications (performance score, price, score per dollar) in the current marketplace is used (as of March 2025). The sample has been take from [Geekbench](https://browser.geekbench.com/mobile-benchmarks), and as for the prices, they are basically the launch prices of smartphones.
- **Model:** A Gaussian Naive Bayes classifier is trained on the data to determine the decision boundary.
- **Visualization:** The decision boundary is plotted, with color intensity reflecting the probability (confidence) of being underpriced or overpriced.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
      git clone https://github.com/your-username/mobile-pricing-portfolio.git
      cd mobile-pricing-portfolio
   ```

2. **Install necessary dependencies:**

   ```bash
      pip install -r requirements.txt
   ```

3. **Run the code:**
   ```bash
      streamlit run app.py
   ```
