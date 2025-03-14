import streamlit as st
import lightgbm
#from ollama import chat
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="SaleSense", layout="wide")

# -----------------------------
# 1) Load models and embeddings
# -----------------------------
# Example:
# word_embedding = joblib.load('path/to/word_embedding.pkl')
# gbm = joblib.load('path/to/gbm.pkl')
#
# If your "Text2Embedding(word_embedding)" is a class, instantiaxte it:
# transform_embedding = Text2Embedding(word_embedding)

# Mock placeholders to illustrate


# Configure the API with your Gemini API key
genai.configure(api_key= st.secrets["api"]["key"])

# Use Gemini 2 Flash
model = genai.GenerativeModel("gemini-2.0-flash")


@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer(
        "dunzhang/stella_en_400M_v5",
        device="cpu",
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
        trust_remote_code=True
    )


def call_gemini(messages):
    """
    Calls Gemini 2 Flash API to improve the product description.
    """
    #user_message = messages[-1]["parts"]  # Get the last user input

    # Get response from Gemini
    response = model.generate_content(messages)

    # Extract only the improved description from the response
    #new_description = response.text.split(':')[-1].strip()

    new_description = response.text.strip()  # Directly get the improved text without splitting

    return new_description

# response = chat(model="llama3.2:3b", messages=messages)
# new_description = response['message']['content'].split(':')[-1]




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
    
#@st.cache_resource

word_embedding =  load_sentence_transformer()


def mock_transform_embedding(text):
    transform_embedding = Text2Embedding(word_embedding)
    embedding = transform_embedding.transform(text)
    return embedding
        
def mock_gbm_predict(embedding):
    # This would actually use your GBM to predict a score
    gbm = lightgbm.Booster(model_file='lightgbm_model_light_we.txt')
    return gbm.predict(embedding) # returns a list with a random float as mock score


# -----------------------------
# 2) Define the iterative logic
# -----------------------------
def improve_description(example, score_threshold=0.9, max_iterations=6):
    # Transform the text into embeddings
    example_embedding = mock_transform_embedding(example)
    best_score = mock_gbm_predict(example_embedding)[0] 
    best_score = float(best_score)

    # Keep track of the first (original) description
    first_example = example
    first_score = best_score
    best_text = example
    print(first_score)
    i = 1
    while best_score < score_threshold and i <= max_iterations:
      #  - The user message includes the current score and the text to improve
        messages = (
        "Your task is to improve the given product description while keeping all key details intact. "
         "You MUST maximize the 'score' by making the description more engaging, informative, and persuasive. "
        "IMPORTANT: Your response should ONLY contain the improved product description. "
        "You are NOT allowed to provide any commentary, explanations, or suggestions—just return the improved text. "
        "The response must not contain any additional information. "
        "dont hallucinate or make up things only base of known information"
        "dont add extra features only improve on desciptiveness"
        "Return ONLY the improved product description and nothing else. Do NOT provide any commentary, context, or services."
        f"Score: {best_score:.2f} | Description: {example}"
        #"reply understood if you understand"
    )


        # 2a) Call your LLM to get improved text

            #response = chat(model="llama3.2:3b", messages=messages)
    
        # Extract the new description from the response
        # Assumes the response content is in the format: "Description: ..."

        #new_description = response['message']['content'].split(':')[-1]

        new_description = call_gemini(messages)



        # 2b) Predict the new score with your GBM
        new_embedding = mock_transform_embedding(new_description)
        new_score = float(mock_gbm_predict(new_embedding)[0])

        # 2c) If better, update best
        print(best_score)
        if new_description and new_score > best_score:
            best_score = new_score
            best_text = new_description

        # Prepare for next iteration
        example = new_description
        i += 1

    # Calculate percent improvement
    percent_of_change = ((best_score - first_score) / first_score) * 100 if first_score != 0 else 0

    return first_example, first_score, best_text, best_score, percent_of_change

def main():
    st.markdown(
        """
        <style>
        .navbar {
            background-color: #6a0dad;
            padding: 10px;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        </style>
        <div class="navbar">A data driven tool for clothing resale descriptions.</div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("SaleSense: Optimizing Clothing Resale")
    st.markdown(
        """
        **Easy to Use ✅** | **Reliable ✅** | **Constructive Feedback ✅**

        Solutions that bring engagement and increase conversions for your listings.

        Elevate your resale business with Salesense, the intelligent tool that __leverages data__ 📊 to craft compelling, accurate descriptions for your clothing 👕 listings. Harness the power of analytics📈 to make every product stand out and drive more sales 🤑.



        """
    )
    # User inputs a product description (or any text)
    user_input = st.text_area(
        "***Enter your original description:***",
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
        if (best_score == original_score):
            st.write("The model encountered an error while please click improve description again(it may have to be clicked more than once)")
        else:
            st.write("**Improved Description:**", best_description)
            st.write("**New Score:**", best_score)
            st.write(f"**Percent Improvement:** {pct_improvement:.2f}%")

if __name__ == "__main__":
    main()


