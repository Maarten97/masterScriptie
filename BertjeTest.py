import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#Code from Copilot.
def load_bertje():
    # Load BERTje model and tokenizer
    model_name = "GroNLP/bert-base-dutch-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def process_question(tokenizer, model, context, question):
    # Encode the context and question
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the answer
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    return answer

def main():
    tokenizer, model = load_bertje()

    # Get context (e.g., a passage of text)
    context = "In het kader van de bescherming van werknemersrechten en het waarborgen van eerlijke arbeidsvoorwaarden, wordt in dit wetsartikel bepaald dat werkgevers verplicht zijn om werknemers te voorzien van een schriftelijke arbeidsovereenkomst. Deze overeenkomst dient alle essentiële elementen van de arbeidsrelatie te bevatten, waaronder de identiteit van de werkgever en werknemer, de functieomschrijving, de arbeidsduur, het salaris, en eventuele specifieke afspraken omtrent vakantiedagen, pensioenregelingen, en andere secundaire arbeidsvoorwaarden. Het is van cruciaal belang dat deze overeenkomst helder en begrijpelijk is voor beide partijen, om misverstanden en geschillen te voorkomen. Bovendien dient de arbeidsovereenkomst te voldoen aan de geldende wet- en regelgeving, inclusief cao-bepalingen indien van toepassing. Werkgevers dienen een exemplaar van de ondertekende arbeidsovereenkomst te verstrekken aan de werknemer, en deze overeenkomst moet op elk moment beschikbaar zijn voor raadpleging door zowel de werkgever als de werknemer. Bij wijzigingen in de arbeidsvoorwaarden is het tevens vereist dat deze schriftelijk worden vastgelegd en door beide partijen worden ondertekend. Het niet naleven van deze verplichtingen kan leiden tot juridische consequenties voor de werkgever, waaronder mogelijke boetes en claims van werknemers. Dit wetsartikel heeft tot doel een transparante en evenwichtige relatie tussen werkgever en werknemer te bevorderen, waarbij de rechten en plichten van beide partijen duidelijk zijn vastgelegd en gerespecteerd worden."

    while True:
        question = input("Stel een vraag (of typ 'exit' om af te sluiten): ")
        if question.lower() == "exit":
            break

        answer = process_question(tokenizer, model, context, question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
