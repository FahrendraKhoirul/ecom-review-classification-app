import torch
import numpy as np
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
from model_services.model import IndoBertEcommerceReview, IndoBertCNNEcommerceReview, IndoBertLSTMEcommerceReview
import streamlit as st


ready_status = False
bert = None
tokenizer = None
indobert_model = None
indobertcnn_model = None
indobertlstm_model = None


with st.status("Loading models...", expanded=True, state='running') as status:
    # Load the base model and tokenizer
    bertSequence = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1",
                                                            num_labels=3,
                                                           problem_type="multi_label_classification")
    bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
    tokenizer = AutoTokenizer.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-reviews")

    # Load custom models
    indobert_model = IndoBertEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-reviews", bert=bertSequence)
    st.write("IndoBERT model loaded")
    indobertcnn_model = IndoBertCNNEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-cnn-finetuned-ecommerce-reviews", bert=bert)
    st.write("IndoBERT-CNN model loaded")
    indobertlstm_model = IndoBertLSTMEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-lstm-finetuned-ecommerce-reviews", bert=bert)
    st.write("IndoBERT-LSTM model loaded")

    # Update status to indicate models are ready
    if indobert_model and indobertcnn_model and indobertlstm_model != None:
        ready_status = True
    if ready_status:
        status.update(label="Models loaded successfully", expanded=False)
        status.success("Models loaded successfully", icon="✅")
    else:
        status.error("Failed to load models")


# def init():
#     global ready_status, bert, tokenizer, indobert_model, indobertcnn_model, indobertlstm_model
#     try:
#         # Load the base model and tokenizer
#         bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
#         tokenizer = AutoTokenizer.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-reviews")
        
#         # Load custom models
#         indobert_model = IndoBertEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-reviews", bert=bert)
#         print("IndoBERT model loaded")
#         indobertcnn_model = IndoBertCNNEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-cnn-finetuned-ecommerce-reviews", bert=bert)
#         print("IndoBERT-CNN model loaded")
#         indobertlstm_model = IndoBertLSTMEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-lstm-finetuned-ecommerce-reviews", bert=bert)
#         print("IndoBERT-LSTM model loaded")
#         ready_status = True
#         return True
#     except Exception as e:
#         print(f"Failed to initialize models: {e}")
#         ready_status = False
#         return False

def predict(text: str, model_name: str):
    token_result = tokenizer(text, return_tensors="pt")
    model = None
    if model_name == "IndoBERT":
        model = indobert_model
    if model_name == "IndoBERT-CNN":
        model = indobertcnn_model
    if model_name == "IndoBERT-LSTM (Best)":
        model = indobertlstm_model
    input_ids = token_result['input_ids']
    attention_mask = token_result['attention_mask']
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return preds

def get_label(preds):
    labels = ["Product", "Customer Service", "Shipping/Delivery"]
    result = [label for i, label in enumerate(labels) if preds[i] > 0.6]
    return result

def get_result(reviews: list[str], model_name: str):
    outputs = []
    for review in reviews:
        preds = predict(review, model_name)
        labels = get_label(preds)
        output = {
            "review": review,
            "predicted_score": preds,
            "predicted_labels": labels
        }
        outputs.append(output)
    return outputs