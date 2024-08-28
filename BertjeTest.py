from huggingface_hub import get_full_repo_name
from transformers import BertTokenizer, BertForPreTraining
import os

# Define the model and tokenizer name
MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_DIR = 'berttrain/mbert'  # Define the directory where you want to save the files

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

# Download and save the model and tokenizer
download_model_and_tokenizer(MODEL_NAME, MODEL_DIR)

print("Download complete!")

# Test function to check if import works
# def test_huggingface_hub():
#     repo_name = "example_repo"
#     full_repo_name = get_full_repo_name(repo_name)
#     print(f"Full repo name: {full_repo_name}")

if __name__ == "__main__":
    download_model_and_tokenizer(model_name=MODEL_NAME, save_directory=MODEL_DIR)


# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#
# #Code from Copilot.
# def load_bertje():
#     # Load BERTje model and tokenizer
#     model_name = "GroNLP/bert-base-dutch-cased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#     return tokenizer, model
#
# def process_question(tokenizer, model, context, question):
#     # Encode the context and question
#     inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#
#     # Get the answer
#     with torch.no_grad():
#         start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
#         answer_start = torch.argmax(start_scores)
#         answer_end = torch.argmax(end_scores) + 1
#         answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
#
#     return answer
#
# def main():
#     tokenizer, model = load_bertje()
#
#     # Get context (e.g., a passage of text)
#     context = "In het kader van de bescherming van werknemersrechten en het waarborgen van eerlijke arbeidsvoorwaarden, wordt in dit wetsartikel bepaald dat werkgevers verplicht zijn om werknemers te voorzien van een schriftelijke arbeidsovereenkomst. Deze overeenkomst dient alle essentiële elementen van de arbeidsrelatie te bevatten, waaronder de identiteit van de werkgever en werknemer, de functieomschrijving, de arbeidsduur, het salaris, en eventuele specifieke afspraken omtrent vakantiedagen, pensioenregelingen, en andere secundaire arbeidsvoorwaarden. Het is van cruciaal belang dat deze overeenkomst helder en begrijpelijk is voor beide partijen, om misverstanden en geschillen te voorkomen. Bovendien dient de arbeidsovereenkomst te voldoen aan de geldende wet- en regelgeving, inclusief cao-bepalingen indien van toepassing. Werkgevers dienen een exemplaar van de ondertekende arbeidsovereenkomst te verstrekken aan de werknemer, en deze overeenkomst moet op elk moment beschikbaar zijn voor raadpleging door zowel de werkgever als de werknemer. Bij wijzigingen in de arbeidsvoorwaarden is het tevens vereist dat deze schriftelijk worden vastgelegd en door beide partijen worden ondertekend. Het niet naleven van deze verplichtingen kan leiden tot juridische consequenties voor de werkgever, waaronder mogelijke boetes en claims van werknemers. Dit wetsartikel heeft tot doel een transparante en evenwichtige relatie tussen werkgever en werknemer te bevorderen, waarbij de rechten en plichten van beide partijen duidelijk zijn vastgelegd en gerespecteerd worden."
#
#     while True:
#         question = input("Stel een vraag (of typ 'exit' om af te sluiten): ")
#         if question.lower() == "exit":
#             break
#
#         answer = process_question(tokenizer, model, context, question)
#         print(f"Answer: {answer}")
#
# if __name__ == "__main__":
#     main()
