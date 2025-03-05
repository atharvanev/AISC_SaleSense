# SaleSense: Optimizing Clothing Resale Descriptions

**SaleSense** is a data-driven tool designed to enhance clothing resale descriptions by improving their quality and accuracy. It leverages machine learning models to generate compelling product descriptions based on user input, maximizing engagement and increasing conversions for clothing listings.

## Features

- **Product Description Improvement**: Iteratively improves product descriptions based on a score threshold.
- **Engaging and Accurate Descriptions**: Uses an LLM model (Ollama's llama3.2) to improve product descriptions while maintaining essential details like sizes, colors, and brands.
- **LightGBM Scoring**: Predicts the quality of descriptions using a LightGBM model.
- **Streamlit Interface**: A user-friendly interface built with Streamlit, making it easy to input descriptions and receive optimized results.

## Installation

### Step 1: Clone the repository

```
git clone https://github.com/yourusername/SaleSense.git
cd SaleSense
```

### Step 2: Install the required libraries

To install the necessary libraries, run the following command:

```
pip install streamlit lightgbm git+https://github.com/ollama/ollama-python.git sentence-transformers
```
### Step 3: Installing the Model Weight

1. **Make sure to download Ollama 3.2:3b**. You can do this by running the following command:

   ```
   ollama pull llama3.2:3b
   ```
### Step 4: Running the App
After installing the required libraries, you can start the Streamlit app by running:
```
streamlit run app.py
```

### Usage

1. Enter your original product description in the provided text area.
2. Click the "Improve Description" button.
3. The app will show:
   - The original description
   - The improved description
   - The percentage improvement in description quality
![Screenshot 2025-03-04 at 7 10 37 PM](https://github.com/user-attachments/assets/0cc8014f-9773-4aaf-bdfd-fe6f94ba5284)

