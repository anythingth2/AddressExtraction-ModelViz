import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import random
import pandas as pd

model = joblib.load('model/model.joblib')

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

def parse_and_visualize(text, selected_entities, highlighted_words, is_initial=False):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predictions = model.predict([features])[0]

    nlp = spacy.blank("th")
    doc = Doc(nlp.vocab, words=tokens)

    colors = {
        "O": "#99ff99",
        "ADDR": "#ffadad",
        "LOC": "#fdffb6",
        "POST": "#9ce7d5"
    }

    entity_counts = Counter(predictions)
    entity_types = ["ADDR", "LOC", "POST", "O"]
    counts = {entity: entity_counts.get(entity, 0) for entity in entity_types}

    html = """
    <style>
    .entity {
        display: inline-block;
        padding: 0.3em 0.6em;
        margin: 0.1em;
        border-radius: 0.5em;
        line-height: 1.2;
        border: 1px solid #ddd;
    }
    .label {
        font-size: 0.8em;
        color: #ffffff;
        padding: 0.2em 0.4em;
        border-radius: 0.3em;
        margin-left: 0.3em;
        vertical-align: middle;
    }
    .empty-border {
        border: 1px solid #ddd;
        background-color: transparent !important;
    }
    .highlighted {
        border: 1px solid transparent;
    }
    </style>
    """
    
    for i, token in enumerate(doc):
        entity_label = predictions[i]
        if is_initial:  # ถ้าเป็นการ analyze ครั้งแรก
            if entity_label in selected_entities:
                color = colors.get(entity_label, "#ffffff")
                html += f"<span class='entity highlighted' style='background-color: {color};'>{token.text}<span class='label' style='background-color: #333;'>{entity_label}</span></span> "
            else:
                html += f"<span class='entity empty-border'>{token.text}</span> "
        else:  # ถ้าเป็นการ shuffle
            if token.text in highlighted_words:
                color = colors.get(entity_label, "#ffffff")
                html += f"<span class='entity highlighted' style='background-color: {color};'>{token.text}<span class='label' style='background-color: #333;'>{entity_label}</span></span> "
            else:
                html += f"<span class='entity empty-border'>{token.text}</span> "

    st.markdown(html, unsafe_allow_html=True)

    # แสดงกราฟ
    if is_initial:  # ถ้าเป็นการ analyze ครั้งแรก แสดงทุก entity
        # fig, ax = plt.subplots(figsize=(6, 3))
        # labels = list(counts.keys())
        # values = list(counts.values())
        # ax.bar(labels, values, color=[colors.get(label, "#ffffff") for label in labels])
        # ax.set_xlabel("Entity Type")
        # ax.set_ylabel("Count")
        # ax.set_title("Entity Count (All Words)")
        # st.pyplot(fig)
        pass
    elif highlighted_words:  # ถ้าเป็นการ shuffle และมีการเลือกคำ
        # selected_predictions = [pred for token, pred in zip(tokens, predictions) if token in highlighted_words]
        # selected_counts = Counter(selected_predictions)
        # counts = {entity: selected_counts.get(entity, 0) for entity in entity_types}
        
        # fig, ax = plt.subplots(figsize=(6, 3))
        # labels = list(counts.keys())
        # values = list(counts.values())
        # ax.bar(labels, values, color=[colors.get(label, "#ffffff") for label in labels])
        # ax.set_xlabel("Entity Type")
        # ax.set_ylabel("Count")
        # ax.set_title("Entity Count (Selected Words)")
        # st.pyplot(fig)
        pass
    return counts


def shuffle_text(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

def create_dataframe_result(data):
    df_result_counter = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df_result_counter.columns = ['Class', 'Count']
    return df_result_counter

st.set_page_config(layout="wide")
# สร้าง UI
st.title("Named Entity Recognition (NER)")

# Input text
text = st.text_input("Text Input:")

# Session state initialization
if 'initial_result' not in st.session_state:
    st.session_state.initial_result = None
if 'shuffled_texts' not in st.session_state:
    st.session_state.shuffled_texts = []

# Sidebar filters
selected_entities = st.sidebar.multiselect(
    "Named Entities",
    options=["ADDR", "LOC", "POST", "O"],
    default=["ADDR", "LOC", "POST", "O"]
)

# Analyze
if st.button("Analyze NER"):
    if text:
        st.write("Initial Analysis of Original Text:")
        st.session_state.initial_result = text
        counts = parse_and_visualize(text, selected_entities, [],is_initial=True)
        st.session_state.ner_done = True
    else:
        st.warning("Please enter text for analysis.")

# Shuffle button and functionality
if 'ner_done' in st.session_state and st.session_state.ner_done:
    if st.button("Shuffle Text"):
        #st.write("Shuffled Texts:")
        st.session_state.shuffled_texts = [shuffle_text(text) for _ in range(5)]
        #for shuffled_text in st.session_state.shuffled_texts:
            #st.text(shuffled_text)
            #st.write('----------------------------------------')
        
        # Enable word selection
        st.session_state.show_word_selection = True

    if hasattr(st.session_state, 'show_word_selection'):
        original_words = text.split()
        highlighted_words = st.sidebar.multiselect(
            "Choose Words to Highlight",
            options=original_words,
            default=[]
        )
        
        
        if highlighted_words or True:  # แสดงทุกครั้งแม้ยังไม่มีการเลือกคำ
            st.write("Visualized Results:")
            for shuffled_text in st.session_state.shuffled_texts:
                counts = parse_and_visualize(shuffled_text, selected_entities, highlighted_words,is_initial=False)
                st.write('----------------------------------------')


else:
    st.write(" ")