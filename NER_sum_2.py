import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np
import pandas as pd

model = joblib.load('model/model.joblib')

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

N_SHUFFLE = 100


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

TAG_COLORS = {
    "O": "#99ff99",
    "ADDR": "#ffadad",
    "LOC": "#fdffb6",
    "POST": "#9ce7d5"
}


TAG_COLORS_VERSION_DEAR = {
    "O": "#FFC0CB",
    "ADDR": "#ADD8E6",
    "LOC": "#fdffb6",
    "POST": "#9ce7d5"
}
def create_token_version_pson(token: str, entity_label: str = None):

    base_html = """
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
    if entity_label is None:
        return base_html + f"<span class='entity empty-border'>{token}</span> "
    else:
        color = TAG_COLORS.get(entity_label, "#ffffff")
        return base_html + f"<span class='entity highlighted' style='background-color: {color};'>{token}<span class='label' style='background-color: #333;'>{entity_label}</span></span> "
    
def create_token_tag_version_dear(token: str, entity_label: str = None) -> str:
    # Define colors for each label and highlighted words
    TAG_COLORS_VERSION_DEAR = {
        "O": "#FFC0CB",
        "ADDR": "#ADD8E6",
        "LOC": "#fdffb6",
        "POST": "#9ce7d5"
    }
    highlight_color = "#FFD700"  # Gold for highlighted words

    color = TAG_COLORS_VERSION_DEAR.get(entity_label, "#ffffff")

    # Initialize HTML output
    html_output = '<div style="font-family: sans-serif; text-align: left; line-height: 1.5;">'

    if entity_label is None:
        html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; padding: 5px;">'
        html_output += f'<div>{token}</div>'
        html_output += '</div>'
    else:
        html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; border: 1px solid {color}; background-color: {color}; padding: 5px; border-radius: 5px;">'
        html_output += f'<div style="color: black; padding: 2px 5px; margin-top: 2px; border-radius: 3px;">{token}</div>'
        html_output += f'<div style="background-color: white; color: #6D6875; padding: 2px 5px; margin-top: 2px; border-radius: 3px; font-weight: bold;">{entity_label}</div>'
        html_output += '</div>'
    
    html_output += '</div>'
    return html_output

create_token_tag = create_token_tag_version_dear
    
def parse_and_visualize(text, selected_entities, highlighted_words, is_initial=False):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predictions = model.predict([features])[0]

    nlp = spacy.blank("th")
    doc = Doc(nlp.vocab, words=tokens)


    entity_counts = Counter(predictions)
    entity_types = ["ADDR", "LOC", "POST", "O"]
    counts = {entity: entity_counts.get(entity, 0) for entity in entity_types}

    n_word = len(doc)
    column_spec = np.full(n_word, 1 / n_word)

    columns = st.columns(column_spec)

    for i, token in enumerate(doc):
        entity_label = predictions[i]
        if is_initial:  # ถ้าเป็นการ analyze ครั้งแรก
            if entity_label in selected_entities:
                token_tag = create_token_tag(token.text, entity_label)
            else:
                token_tag = create_token_tag(token.text)
        else:  # ถ้าเป็นการ shuffle
            if token.text in highlighted_words:
                token_tag = create_token_tag(token.text, entity_label)
            else:
                token_tag = create_token_tag(token.text)

        columns[i].markdown(token_tag, unsafe_allow_html=True)
        
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
text = st.text_input("Text Input:", value=None)

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



tag_selector_section, token_selector_section = st.columns(
    spec=[0.2, 0.8]
)

with tag_selector_section:
    selected_entities = st.pills(
        'Named Entities',
        options=["ADDR", "LOC", "POST", "O"],
        default=["ADDR", "LOC", "POST", "O"],
        selection_mode='multi'
    )

# with token_selector_section:

# Analyze
# if st.button("Analyze NER"):
if text:
    st.session_state.is_analyzed = True

if 'is_analyzed' in st.session_state and st.session_state.is_analyzed:
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
        st.session_state.shuffled_texts = [shuffle_text(text) for _ in range(N_SHUFFLE)]
        #for shuffled_text in st.session_state.shuffled_texts:
            #st.text(shuffled_text)
            #st.write('----------------------------------------')
        
        # Enable word selection
        st.session_state.show_word_selection = True

    if hasattr(st.session_state, 'show_word_selection'):
        original_words = text.split()
        
        st.sidebar.multiselect(
            "Choose Words to Highlight",
            options=original_words,
            default=[]
        )

        with token_selector_section:
            highlighted_words = st.pills(
                'Choose Words to Highlight',
                options=original_words,
                default=[],
                selection_mode='multi'
            )
        highlighted_words = highlighted_words or list()
        
        if highlighted_words or True:  # แสดงทุกครั้งแม้ยังไม่มีการเลือกคำ
            st.write("Visualized Results:")
            print('selected_entities', selected_entities)
            print('highlighted_words', highlighted_words)
            for shuffled_text in st.session_state.shuffled_texts:
                counts = parse_and_visualize(shuffled_text, selected_entities, highlighted_words,is_initial=False)
                st.write('----------------------------------------')


else:
    st.write(" ")