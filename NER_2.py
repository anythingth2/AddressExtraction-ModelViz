import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import random
import pandas as pd


model = joblib.load('/Users/aye_/Documents/VS/model.joblib')

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

# ฟังก์ชันสร้างฟีเจอร์ให้แต่ละ token
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

def parse_and_visualize(text, selected_entities):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predictions = model.predict([features])[0]

    # สร้าง spaCy Doc object
    nlp = spacy.blank("th")
    doc = Doc(nlp.vocab, words=tokens)

    # กำหนดสีสำหรับการแสดงผลแต่ละประเภท
    colors = {
        "O": "#ffffff",     # ไม่มีเอนทิตี้
        "ADDR": "#ffadad",  # ที่อยู่
        "LOC": "#fdffb6",   # ตำแหน่ง
        "POST": "#9ce7d5"   # รหัสไปรษณีย์
    }

    # นับจำนวนเอนทิตี้แต่ละประเภท
    entity_counts = Counter(predictions)

    # กำหนดลำดับของประเภท Entity ที่ต้องการแสดงในกราฟ
    entity_types = ["ADDR", "LOC", "POST"]

    # ตรวจสอบว่าแต่ละประเภทมีค่าหรือไม่ ถ้าไม่มีให้ใส่ 0
    counts = {entity: entity_counts.get(entity, 0) for entity in entity_types}

    # สร้าง HTML พร้อมปรับแต่งให้สวยงามสำหรับการแสดงใน Streamlit
    html = """
    <style>
    .entity {
        display: inline-block;
        padding: 0.3em 0.6em;
        margin: 0.1em;
        border-radius: 0.5em;
        line-height: 1.2;
    }
    .label {
        font-size: 0.8em;
        color: #ffffff;
        padding: 0.2em 0.4em;
        border-radius: 0.3em;
        margin-left: 0.3em;
        vertical-align: middle;
    }
    </style>
    """

    # เพิ่มแต่ละคำพร้อมกับกล่องไฮไลต์และแสดงประเภท Entity
    for i, token in enumerate(doc):
        entity_label = predictions[i]
        
        # กำหนดสีให้กับคำที่เลือกจากฟิลเตอร์
        if entity_label in selected_entities:
            color = colors.get(entity_label, "#ffffff")
            html += f"<span class='entity' style='background-color: {color};'>{token.text}<span class='label' style='background-color: #333;'>{entity_label}</span></span> "
        else:
            # ถ้าคำนี้ไม่อยู่ในประเภทที่เลือก ให้ไม่แสดงสีพื้นหลัง
            html += f"<span class='entity' style='background-color: transparent;'>{token.text}</span> "

    # ใช้ columns ในการจัดรูปแบบใน Streamlit
    col1, col2 = st.columns([2, 1])  # กำหนด col1 มีพื้นที่มากกว่า col2

    # แสดงผล HTML ใน Streamlit
    with col1:
        st.markdown(html, unsafe_allow_html=True)

    # สร้างกราฟแท่งสรุปประเภท Entity
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(6, 3))  # ปรับขนาดกราฟให้เล็กลง
    ax.bar(labels, values, color=[colors.get(label, "#ffffff") for label in labels])
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    ax.set_title("Entity Count ")

    # แสดงกราฟในคอลัมน์ด้านขวา
    with col2:
        st.pyplot(fig)

# ฟังก์ชันแสดงผลการ shuffle ข้อความ
def shuffle_text(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

# ฟังก์ชันสร้าง DataFrame จาก Counter
def create_dataframe_result(data):
    # แปลงจาก Counter เป็น DataFrame โดยไม่แสดง Error หรือผลลัพธ์ที่ผิดพลาด
    df_result_counter = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df_result_counter.columns = ['Class', 'Count']
    return df_result_counter

# สร้าง UI ใน Streamlit
st.title("Named Entity Recognition (NER)")

# Input text สำหรับการทำนาย
text = st.text_input("Text Input:")

# ตัวแปรสำหรับเก็บผลการวิเคราะห์ตั้งต้น
if 'initial_result' not in st.session_state:
    st.session_state.initial_result = None

# เพิ่มตัวเลือก filter ที่แถบเครื่องมือด้านซ้าย
selected_entities = st.sidebar.multiselect(
    "Named Entities",
    options=["ADDR", "LOC", "POST"],
    default=["ADDR", "LOC", "POST"]
)

# ปุ่มวิเคราะห์ข้อความ
if st.button("Analyze NER"):
    st.session_state.initial_result = text  # เก็บข้อความตั้งต้น
    parse_and_visualize(text, selected_entities)  # วิเคราะห์ข้อความตั้งต้น

# ปุ่ม Shuffle ข้อความ
if st.button("Shuffle Text"):
    shuffled_text = shuffle_text(text)
    st.write(f"Shuffled Text: {shuffled_text}")
    # วิเคราะห์ข้อความที่สลับแล้ว
    parse_and_visualize(shuffled_text, selected_entities)  
    st.write('----------------------------------------')

    # แสดงผลข้อความต้นฉบับหลังจาก shuffle
    if st.session_state.initial_result is not None:
        st.write("Initial Analysis of Original Text:")
        parse_and_visualize(st.session_state.initial_result, selected_entities)
