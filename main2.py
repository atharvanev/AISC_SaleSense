import streamlit as st
import random
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1) Load models and embeddings
# -----------------------------
class Text2Embedding():
    def __init__(self, model):
        self.model = model

    def sentence2vector(self, sentences, name='product_description'):
        vector = self.model.encode(sentences)
        return vector

    def transform(self, input):
        embedding = self.sentence2vector(input)
        return embedding.reshape(1, 1024)

# Initialize embedding model
word_embedding = SentenceTransformer("dunzhang/stella_en_400M_v5", device="cpu", config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}, trust_remote_code=True)

# -----------------------------
# 2) Mock Prediction Function
# -----------------------------
def mock_gbm_predict(embedding):
    # Simulate the prediction, return a random float as score
    return [round(random.uniform(0.5, 0.9), 2)]  # Mock score based on embedding

# -----------------------------
# 3) Define the iterative logic
# -----------------------------
def improve_description(example, score_threshold=0.9, max_iterations=6):
    transform_embedding = Text2Embedding(word_embedding)
    example_embedding = transform_embedding.transform(example)

    # Initial score prediction
    best_score = mock_gbm_predict(example_embedding)[0]
    best_score = float(best_score)

    # Track original values
    first_example = example
    first_score = best_score
    best_text = example

    i = 1
    while best_score < score_threshold and i <= max_iterations:
        # Simulate improvement in description (since no API call)
        new_description = best_text + " (improved)"  # Just add "(improved)" to simulate change

        # Simulate new score prediction
        new_embedding = transform_embedding.transform(new_description)
        new_score = float(mock_gbm_predict(new_embedding)[0])

        # If the new description has a higher score, update the best description
        if new_score > best_score:
            best_score = new_score
            best_text = new_description

        example = new_description
        i += 1

    # Calculate percent improvement
    percent_of_change = ((best_score - first_score) / first_score) * 100 if first_score != 0 else 0

    return first_example, first_score, best_text, best_score, percent_of_change

def main():
    st.title("Iterative Text Improvement Demo")

    # User inputs a product description (or any text)
    user_input = st.text_area(
        "Enter your original description:",
        value=(
            "Authentic vintage Chanel made out of luxurious black lambskin. "
            "Featuring gold CC closure. Size W: 25cm H: 17cm Size D: 2cm. "
            "Shoulder height: 94cm. Inside lining has been fully replaced. Comes with ribbon."
        ),
        height=150
    )

    # Button to trigger improvement
    if st.button("Improve Description"):
        with st.spinner("Improving description..."):
            # Run the iterative improvement
            original_description, original_score, best_description, best_score, pct_improvement = improve_description(user_input)

        # Display results
        st.subheader("Results")
        st.write("**Original Description:**", original_description)
        st.write("**Original Score:**", original_score)
        st.write("**Improved Description:**", best_description)
        st.write("**Best Score:**", best_score)
        st.write(f"**Percent Improvement:** {pct_improvement:.2f}%")

if __name__ == "__main__":
    main()
