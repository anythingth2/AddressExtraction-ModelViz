# %%
import pandas as pd


from IPython.display import display

# %%
# !pip install sklearn_crfsuite
import joblib
model = joblib.load("model/model.joblib")

# %%
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

def parse(text):
	tokens = text.split()
	
	features = [tokens_to_features(tokens, i) for i in range(len(tokens))]

	output = model.predict([features])[0]
	return tokens, output

# %%
parse("นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330")

# %%
parse("นายมงคล 123/4 ตำบล บ้านไกล อำเภอ เมือง จังหวัด ลพบุรี 15000")

# %%
text = "นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330"
tokens = text.split()

# %%
features = [tokens_to_features(tokens, i) for i in range(len(tokens))]

# %%
feature_df = pd.DataFrame(features)
display(feature_df)

# %%


# %% [markdown]
# # Streamlit

# %%
import streamlit as st


# %%
default_address_text = 'นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330'

# %%
st.title('Address Extraction')
address_text = st.text_input('Address', default_address_text)

if st.button('Submit'):
	tokens, output = parse(address_text)
	output_df = pd.DataFrame({
		'token': tokens,
		'output': output
	})
	st.write(output_df.T)


# %%



