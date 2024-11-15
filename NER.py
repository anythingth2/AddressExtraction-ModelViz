import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import random

# Load the model
try:
    model = joblib.load("model/model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }

    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features

def parse_and_visualize(text, selected_entities, highlighted_words=None):
    try:
        tokens = text.split()
        features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
        
        if model is None:
            raise ValueError("Model is not loaded.")
        
        predictions = model.predict([features])[0]

        # Define colors for each label and highlighted words
        label_colors = {
            "O": "#FFC0CB",
            "ADDR": "#ADD8E6",
            "LOC": "#fdffb6",
            "POST": "#9ce7d5"
        }
        highlight_color = "#FFD700"  # Gold for highlighted words

        # Initialize HTML output
        html_output = '<div style="font-family: sans-serif; text-align: left; line-height: 1.5;">'
        
        for i, token in enumerate(tokens):
            label = predictions[i]
            label_color = label_colors.get(label, "#ffffff")  # Default to white if no color is assigned
            
            # Set color to highlight_color if the token is in highlighted_words
            color = highlight_color if highlighted_words and token in highlighted_words else label_color

            if label in selected_entities:
                html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; border: 1px solid {color}; background-color: {color}; padding: 5px; border-radius: 5px;">'
                html_output += f'<div style="color: black; padding: 2px 5px; margin-top: 2px; border-radius: 3px;">{token}</div>'
                html_output += f'<div style="background-color: white; color: #6D6875; padding: 2px 5px; margin-top: 2px; border-radius: 3px; font-weight: bold;">{label}</div>'
                html_output += '</div>'
            else:
                html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; padding: 5px;">'
                html_output += f'<div>{token}</div>'
                html_output += '</div>'
        
        html_output += '</div>'
        
        # Display the final HTML output with highlighted colors
        return html_output

    except Exception as e:
        st.error(f"NER processing error: {e}")
        return ""


def shuffle_text(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

# Set page layout to wide
st.set_page_config(layout="wide")

st.title("Named Entity Recognition (NER)")

# Text area input for user
text_input = st.text_area("Enter text here:", "")

# Function to update NER output based on selected entities
def update_all_outputs():
    if text_input:
        # Update both main NER output and shuffled outputs when entities are changed
        st.session_state.ner_output = parse_and_visualize(text_input, st.session_state.selected_entities)
        st.session_state.shuffled_outputs = [
            parse_and_visualize(shuffle_text(text_input), st.session_state.selected_entities)
            for _ in range(5)
        ]

# Custom CSS for sidebar styling
st.markdown(
    """
    <style>
    /* Change sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6; /* Choose your desired background color */
    }

    /* Change multiselect tag background color and text color */
    [data-baseweb="tag"] {
        background-color: #7bbff5 !important; /* Your desired tag color */
        color: #ffffff !important; /* Text color */
    }

    /* Change multiselect tag close button color */
    [data-baseweb="tag"] svg {
        fill: #ffffff !important; /* Close icon color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Entity filter in the sidebar with on_change callback to update both outputs
st.sidebar.multiselect(
    "Named Entities",
    options=["ADDR", "LOC", "POST", "O"],
    default=["ADDR", "LOC", "POST", "O"],
    key="selected_entities",
    on_change=update_all_outputs
)

# Adjusted function to update main output with highlighted words after shuffle
def update_highlighted_words_output():
    if text_input:
        st.session_state.ner_output = parse_and_visualize(text_input, st.session_state.selected_entities, st.session_state.highlighted_words)


# Modify the shuffle logic to show the highlighted words selection after shuffling
with st.container():
    if st.button("Analyze") and text_input:
        st.session_state.ner_output = parse_and_visualize(text_input, st.session_state.selected_entities)
        st.session_state.show_shuffle = True  # Enable Shuffle button

    if st.session_state.get("show_shuffle", False):
        if st.button("Shuffle"):
            st.session_state.shuffled_outputs = [
                parse_and_visualize(shuffle_text(text_input), st.session_state.selected_entities)
                for _ in range(5)
            ]
            st.session_state.show_shuffled_outputs = True  # Enable display of shuffled outputs and highlighted words selection

# Display main NER output and shuffled text outputs
if "ner_output" in st.session_state:
    st.markdown(st.session_state.ner_output, unsafe_allow_html=True)
    st.write("-------")

# Display the shuffled texts if Shuffle button has been pressed
if st.session_state.get("show_shuffled_outputs", False):
    for i, shuffled_output in enumerate(st.session_state.shuffled_outputs, 1):
        st.write(f"Shuffled Text {i}:")
        st.markdown(shuffled_output, unsafe_allow_html=True)
        st.write("-------")

    # Sidebar selection for highlighted words (shown only after shuffle)
    original_words = text_input.split()
    highlighted_words = st.sidebar.multiselect(
        "Choose Words to Highlight",
        options=original_words,
        default=original_words,
    key="highlighted_words",
    on_change=update_highlighted_words_output
)