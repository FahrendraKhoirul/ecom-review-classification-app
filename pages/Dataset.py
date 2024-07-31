# streamlit
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

st.set_page_config(page_title="Dataset", page_icon="🏠")

# help me create sidebar
st.sidebar.markdown("## 📚 About"
                    "\nThis is a simple web app to classify the aspect of reviews from an e-commerce dataset."
                    "\n\nThe dataset used is a multilabel dataset, which means a review can have multiple labels."
                    "\n\nThe labels are:"
                    "\n- 📦 **Product**"
                    "\n- 👩‍💼 **Customer Service**"
                    "\n- 🚚 **Shipping/Delivery**")

# add create by Fahrendra Khoirul Ihtada and Rizha Alfianita using streamlit and Hugging Face's IndoBERT model
st.sidebar.markdown("## 👨‍💻 Created by"
                    "\n[Fahrendra Khoirul Ihtada](https://www.linkedin.com/in/fahrendra-khoirul-ihtada/) "
                    "and [Rizha Alfianita](https://www.linkedin.com/in/rizha-alfianita/)"
                    "\n Using Streamlit and Hugging Face's [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) model.")

# add my hugging face profile
st.sidebar.markdown("## 🤖 Hugging Face"
                    "\n- [Fahrendra Khoirul Ihtada](https://huggingface.co/fahrendrakhoirul)")

# Title and Caption
st.title("📊Dataset Overview")

# Descriptive text
st.write("""
This dataset is full of customer reviews that give us a great idea of what it's like to buy things online. The reviews talk about everything from how good the product is to how fast it got here and how helpful the seller was""")

# Dataset link
st.markdown("[Access the Dataset](https://huggingface.co/datasets/fahrendrakhoirul/ecommerce-reviews-multilabel-dataset)")
 

df = pd.read_json("Product Reviews Ecommerce Multilabel Dataset.json", lines=True)
st.write(df)

# # Combine all reviews into a single string
# all_reviews = " ".join(df['review'])   


# # Generate the word cloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

# # Display the word cloud
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show() 

# st.pyplot(plt.gcf())

# # Create a bar chart for sentiment analysis (example)
# # Assuming you have a column 'sentiment' in your dataframe
# # st.write("**Distribusi Sentimen**")
# # sentiment_counts = df['sentiment'].value_counts()
# # st.bar_chart(sentiment_counts)
