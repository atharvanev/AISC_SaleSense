import streamlit as st
import lightgbm
from ollama import chat
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1) Load models and embeddings
# -----------------------------
# Example:
# word_embedding = joblib.load('path/to/word_embedding.pkl')
# gbm = joblib.load('path/to/gbm.pkl')
#
# If your "Text2Embedding(word_embedding)" is a class, instantiate it:
# transform_embedding = Text2Embedding(word_embedding)

# Mock placeholders to illustrate
class Text2Embedding():
    def __init__(self, model):
        """
        The constructor takes in a text-embedding model object (e.g., 
        a SentenceTransformer instance or any other encoding model).
        
        :param model: A text embedding model with a .encode() method.
        """
        self.model = model

    def sentence2vector(self, sentences, name='product_description'):
        """
        Encodes a given text (or list of texts) into an embedding.
        
        :param sentences: The text input(s) to encode.
        :param name: Optional identifier for debugging/logging purposes.
        :return: The raw embedding vector (or a list of vectors).
        """
        print(f"Encoding sentences for feature '{name}'...")
        vector = self.model.encode(sentences)
        return vector

    def transform(self, input):
        """
        A convenience method that calls sentence2vector() and reshapes 
        the resulting embedding(s) to a fixed shape of (1, 1024). 
        This implies the model outputs a 1024-dimensional embedding.
        
        :param input: The text input to encode.
        :return: A (1, 1024) NumPy array representing the single text embedding.
        """
        embedding = self.sentence2vector(input)
        return embedding.reshape(1, 1024)
    
@st.cache_resource
def load_model():
    # Load the model (e.g., from Hugging Face)
    model = SentenceTransformer(
        "dunzhang/stella_en_1.5B_v5",
        device="mps",  # Use Metal Performance Shaders on Apple Silicon
        config_kwargs={"use_memory_efficient_attention": False},  # Optional settings
        trust_remote_code=True  # Allow remote execution of model code
    )
    return model


def mock_transform_embedding(text):
    transform_embedding = Text2Embedding(load_model())
    embedding = transform_embedding.transform(text)
    return embedding
        
def mock_gbm_predict(embedding):
    # This would actually use your GBM to predict a score
    gbm = lightgbm.Booster(model_file='model_gbm.txt')
    return gbm.predict(embedding) # returns a list with a random float as mock score

def mock_llm_api_call(messages):
    """
    Placeholder function for your real LLM API.
    In production, replace with your actual call to llama3.2 or other API.
    """
    user_message = [m for m in messages if m["role"] == "user"][0]["content"]
    desc = user_message.split("Description:")[-1].strip()
    return {
        "content": "Description: " + desc + " (improved)"
    }

# -----------------------------
# 2) Define the iterative logic
# -----------------------------
def improve_description(example, score_threshold=0.9, max_iterations=3):
    # Transform the text into embeddings
    example_embedding = mock_transform_embedding(example)
    best_score = mock_gbm_predict(example_embedding)[0] 
    best_score = float(best_score)

    # Keep track of the first (original) description
    first_example = example
    first_score = best_score
    best_text = example

    i = 1
    while best_score < score_threshold and i <= max_iterations:
        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to improve the description based on the 'score'. "
                    "Try to maximize 'score' which indicates how good the description is. "
                    "Keep all essential information like sizes, colors, brand, etc. "
                    "Only return the 'Description' text."
                )
            },
            {
                "role": "user",
                "content": f"Score: {best_score:.2f} | Description: {example}"
            },
            {
                "role": "assistant",
                "content": f"Previous response: Score: {first_score:.2f} | Description: {first_example}"
            }
        ]

        # 2a) Call your LLM to get improved text
        response = mock_llm_api_call(messages)
        new_description = response["content"].split("Description:")[-1].strip()

        # 2b) Predict the new score with your GBM
        new_embedding = mock_transform_embedding(new_description)
        new_score = float(mock_gbm_predict(new_embedding)[0])

        # 2c) If better, update best
        if new_score > best_score:
            best_score = new_score
            best_text = new_description

        # Prepare for next iteration
        example = new_description
        i += 1

    # Calculate percent improvement
    percent_of_change = ((best_score - first_score) / first_score) * 100 if first_score != 0 else 0

    return first_example, first_score, best_text, best_score, percent_of_change

# -----------------------------
# 3) Build the Streamlit UI
# -----------------------------
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
