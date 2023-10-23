import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from azure.storage.blob import BlobServiceClient
from flask import Flask, request, jsonify

app = Flask(__name__)

# BERT model and tokenizer
model_name = "textattack/bert-base-uncased-yelp-polarity"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict the category
def predict_category(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = logits.softmax(dim=1)
    predicted_category = ["Documentation", "Content", "Memes"][torch.argmax(probabilities)]
    return predicted_category

# Function to extract text from JSON and predict the category
def predict_category_from_json(json_data):
    input_text = json_data.get('text', '')
    category = predict_category(input_text)
    return category

# Importing data from blob storage
def import_data_from_blob(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    content = blob_data.readall()
    return content

@app.route('/predict_category', methods=['POST'])
def predict_category_api():
    try:
        # Assuming JSON format with a key named 'text' that contains the text data.
        json_data = request.get_json()
        input_text = json_data.get('text', '')

        # Predict the category
        category = predict_category(input_text)

        response = {'category': category}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Azure Blob Storage connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=keywisestorage;AccountKey=uRzlCQwv/SSF6WgkEz0g83dBjnFrziSNNt8PIY5Nnt+OJic0v5xjPnO8ZMhb7SjyesYSOK79TbJ/+AStdLKiDw==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Define your container and blob name
    container_name = "keywisestorage"
    blob_name = "pagescontainer"

    app.run(host="0.0.0.0", port=5000)
