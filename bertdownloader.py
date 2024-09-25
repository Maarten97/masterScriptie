from transformers import BertTokenizer, BertForPreTraining, RobertaTokenizer, RobertaForSequenceClassification
import os

# Define the model and tokenizer name
MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_DIR = 'bert/mbert'  # Define the directory where you want to save the files

# Create the directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


def download_model_and_tokenizer(model_name, save_directory):
    # Download and save the tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print(f"Tokenizer saved to {save_directory}")

    # Download and save the model
    print(f"Downloading model for {model_name}...")
    model = BertForPreTraining.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")


def roberta_tokenizer(model_name, save_directory):
    # Download and save the tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print(f"Tokenizer saved to {save_directory}")

    # Download and save the model
    print(f"Downloading model for {model_name}...")
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")


if __name__ == "__main__":
    download_model_and_tokenizer(model_name=MODEL_NAME, save_directory=MODEL_DIR)
    # roberta_tokenizer(model_name=MODEL_NAME, save_directory=MODEL_DIR)
