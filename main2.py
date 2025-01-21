import streamlit as st

import lightgbm
from ollama import chat
from sentence_transformers import SentenceTransformer


# Create a SentenceTransformer instance with the stella_en_1.5B_v5 model
word_embedding = SentenceTransformer(
    "dunzhang/stella_en_1.5B_v5",     # The name/identifier of the Hugging Face model 
    device="mps",                     # Use Apple's Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon
    config_kwargs={"use_memory_efficient_attention": False},  # Disable memory-efficient attention to ensure compatibility
    trust_remote_code=True            # Allow execution of custom code from the model's repository
)

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


# Load the model from 'model_gbm.txt' file

gbm = lightgbm.Booster(model_file='model_gbm.txt')


# Define the initial description text
example = "Authentic vintage Chanel made out of luxurious black lambskin. Featuring gold CC closure. Size W: 25cm H: 17cm Size D: 2cm. Shoulder height: 94cm. Inside lining has been fully replaced. Comes with ribbon."

# Transform the text into an embedding 
transform_embedding = Text2Embedding(word_embedding)
example_embedding = transform_embedding.transform(example)

# Define a threshold for the score and the maximum number of iterations
score_threshold = 0.9
max_iterations = 3

# Use a GBM model to predict the "score" for the current description
best_score = gbm.predict(example_embedding)
best_score = float(best_score[0])

# Keep track of the first (original) description and its score
first_example = example
first_score = best_score

# Start iterating until we meet our threshold or reach the max_iterations
i = 1
while best_score < score_threshold and i <= max_iterations:
    # Prepare the prompt as messages for the chat model
    #  - The system message instructs the model to improve the description
    #  - The user message includes the current score and the text to improve
    messages = [
        {
            "role": "system",
            "content": (
                "You task is to improve the description based on the 'score' , "
                "try to maximize 'score' which indicates how good is the description, "
                "also keep all the essential information like (sizes, colors, brand, etc). "
                "Note just return the 'Description'"
            )
        },
        {
            "role": "user",
            "content": f"Score: {best_score:.2f} | Description: {example} "
        },
    ]

    # Call the chat model to get a new/improved description
    response = chat(model="llama3.2", messages=messages)
    
    # Extract the new description from the response
    # Assumes the response content is in the format: "Description: ..."
    example = response['message']['content'].split(':')[-1]

    # Transform the new description to get its embedding
    example_embedding = transform_embedding.transform(example)
    
    # Predict the score of this new description
    new_score = gbm.predict(example_embedding)
    new_score = float(new_score[0])

    # Print debug information: how the score changed from old to new
    print(f"Iteration {i}: Old Score={best_score:.4f}, New Score={new_score:.4f}")

    # If the new score is better, update our best_score and best_text
    
    if new_score > best_score:
        best_score = new_score
        best_text = example

    i += 1  # Move to the next iteration

# Once we exit the loop, track the final score
last_score = best_score

# Calculate the percentage improvement over the first/original score
percent_of_change = (((last_score - first_score) / first_score) * 100)

# Print out the old and best descriptions with their corresponding scores
print(f'\nOld description: {first_example}')
print(f"\nBest description: {best_text}")
print(f"\nBest score: {best_score}")
print(f"\nPercent of improvment: {percent_of_change:.3f}%")

st.write(best_text)