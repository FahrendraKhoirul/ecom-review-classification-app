# streamlit
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")

st.markdown("# 🛍️ Aspect-Based Multilabel Classification of Ecommerce Reviews")
st.write("Ever wondered what people think about the products, customer service, and shipping of your favorite online store? Try this out!")

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



# import here because why not??
import model_services.pipeline as pipeline

container_1 = st.container(border=True)


# create rows and 2 dropdown menus side by side
row1_1, row1_2 = container_1.columns((2, 1))
with row1_1:
    df = pd.read_json("Product Reviews Ecommerce Multilabel Dataset.json", lines=True)
    selected_review = st.selectbox(
        "You can pick a review from dataset",
        df["review"].values,
    )
with row1_2:
    selected_model = st.selectbox(
        "Choose the model",
        ("IndoBERT", "IndoBERT-CNN (Best)", "IndoBERT-LSTM"),
    )

# text input
input_review = container_1.text_area("Or you can input multiple review with separated line", selected_review, height=200)

# create button submit
button_submit = container_1.button("Classify")


def show_label_desc():
    st.divider()
    st.write("Let's see what is the meaning of each labels:")
    st.write("- 📦**Product**         : related Customer satisfaction with the quality, performance, and conformity of the product to the description given")
    st.write("- 👩‍💼**Customer Service**  : Interaction between customers and sellers, friendliness and speed of response from sellers, and handling complaints.")
    st.write("- 🚚**Shipping/Delivery** : related to shipping speed, condition of goods when received, and timeliness of shipping")

def submit():
    # Create UI for Result
    st.success("Done! 👌")
    outputs = do_calculation(input_review)
    # input_review = ""
    show_result(outputs)
    show_label_desc()

def do_calculation(texts):
    # split text by newline
    reviews = texts.split("\n")
    # remove empty string
    reviews = list(filter(None, reviews))
    # do the prediction
    outputs = pipeline.get_result(reviews, selected_model)
    return outputs

st.markdown("""
    <style>
    .label-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .rounded-label-product {
        background-color: #FFD700;
        color: black;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 14px;
        margin-bottom: 20px;
    }
            
    .rounded-label-customer-service {
        background-color: #FFA07A;
        color: black;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 14px;
        margin-bottom: 20px;
    }
            
    .rounded-label-shipping-delivery {
        background-color: #20B2AA;
        color: black;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 14px;
        margin-bottom: 20px;
    }
            
    .rounded-label-undefined {
        background-color: #DCDCDC;
        color: black;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 14px;
        margin-bottom: 20px;
    }
            
    </style>
    """, unsafe_allow_html=True)

def chips_label(output):
    asd = []
    for label in output["predicted_labels"]:
        if label == "Product":
            score = f"{output['predicted_score'][0] * 100:.2f}%"
            score = f"<strong>{score}</strong>"
            asd.append(f"<div class='rounded-label-product'>📦Product {score}</div>")
        elif label == "Customer Service":
            score = f"{output['predicted_score'][1] * 100:.2f}%"
            score = f"<strong>{score}</strong>"
            asd.append(f"<div class='rounded-label-customer-service'>👩‍💼Customer Service {score}</div>")
        elif label == "Shipping/Delivery":
            score = f"{output['predicted_score'][2] * 100:.2f}%"
            score = f"<strong>{score}</strong>"
            asd.append(f"<div class='rounded-label-shipping-delivery'>🚚Shipping/Delivery {score}</div>")
    # for label, score in zip(output["predicted_labels"], output["predicted_score"]):
    #     score = f"{score * 100:.2f}%"
    #     score = f"<strong>{score}</strong>"
    #     if label == "Product":
    #         asd.append(f"<div class='rounded-label-product'>📦Product {score}</div>")
    #     elif label == "Customer Service":
    #         asd.append(f"<div class='rounded-label-customer-service'>👩‍💼Customer Service {score}</div>")
    #     elif label == "Shipping/Delivery":
    #         asd.append(f"<div class='rounded-label-shipping-delivery'>🚚Shipping/Delivery {score}</div>")
    if asd == []:
            asd.append("<div class='rounded-label-undefined'>Undefined</div>")
    labels_html = "".join(asd)
    st.markdown(f"<div class='label-container'>{labels_html}</div>", unsafe_allow_html=True)

def show_result(outputs):
    st.title("Result")
    # create 2 column
    col1, col2 = st.columns(2)
    with col1:
        st.write("📑 Total reviews   : ", len(outputs))
    with col2:
        st.write("🖥️ Model used      : ", selected_model)
    for i, output in enumerate(outputs):
        st.markdown(
        f"<p style='color:grey; margin: 0; padding: 0;'>Review {i+1}:</p>", 
        unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px; margin-bottom: 5px;'><strong>{output['review']}</strong></p>", unsafe_allow_html=True)
        chips_label(output)
    st.balloons()
    # change predicted_labels to dict with key is the label
    new_outputs = []
    for output in outputs:
        temp = output
        temp['predicted_score'] = [
            f"Product {output['predicted_score'][0] * 100:.2f}%",
            f"Customer Service {output['predicted_score'][1] * 100:.2f}%",
            f"Shipping/Delivery {output['predicted_score'][2] * 100:.2f}%"
        ]
        new_outputs.append(temp)
    
    df = pd.DataFrame(new_outputs)
    st.write(df)

    # create note if wanna download, hove on top right in table show
    st.markdown("**Note:** To download the table, hover over the top right corner of the table and click the download button.")
    



if button_submit and pipeline.ready_status:
    submit()
elif button_submit and not pipeline.ready_status:
    st.error("Models are not ready yet, please wait a moment")

    