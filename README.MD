# Mobile Pricing Decision Boundary Project

## Overview
[Plot Preview](https://smartphone-pricing.streamlit.app) 

This project uses the Logistic Regression classifier to help mobile manufacturers price new phones based on performance(CPU, GPU, and CPU/RAM interaction) and the current market conditions. An interactive visualization shows a decision boundary, with color intensity representing the classifier's confidence:

- **Deeper green:** Higher confidence that a phone is underpriced.
- **Deeper red:** Higher confidence that a phone is overpriced.

## Dataset & Methodology

- **Dataset:** A sample set of smartphone specifications brought from scraping GSM Arena, smartphones are from the beginning of 2024 to May 2025. The most significant feature that affects the price has been extracted, which turned out to be the performance score from Aututu.
- **Model:** A Logistic Regression classifier is trained on the data to determine the decision boundary.
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
