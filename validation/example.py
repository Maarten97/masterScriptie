# Import necessary libraries
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the tokenizer for multilingual BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load only the Dutch part of the MultiEURLEX dataset
dataset = load_dataset('coastalcph/multi_eurlex', 'nl')

# Tokenize the Dutch dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=32)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels to a fixed-size tensor for multi-label classification
def multi_hot_encode_labels(batch):
    num_labels = len(dataset['train'].features['labels'].feature.names)
    encoded_labels = torch.zeros((len(batch['labels']), num_labels), dtype=torch.float32)
    for i, labels in enumerate(batch['labels']):
        encoded_labels[i, labels] = 1.0  # Multi-hot encoding
    batch['labels'] = encoded_labels
    return batch

tokenized_datasets = tokenized_datasets.map(multi_hot_encode_labels, batched=True)

# Set up the train, validation, and test splits
train_dataset = tokenized_datasets['train'].shuffle(seed=42)
eval_dataset = tokenized_datasets['validation']
test_dataset = tokenized_datasets['test']

# Initialize the model for multi-label classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(dataset['train'].features['labels'].feature.names)
)
model.to(device)

# Load the metric for evaluation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()  # Sigmoid + threshold for multi-label
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Custom Trainer to use BCEWithLogitsLoss with correct label type
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()  # Ensure labels are of type Float
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed `evaluation_strategy` to `eval_strategy`
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
trainer.evaluate(eval_dataset=test_dataset)

# Example prediction on a new Dutch text
sample_text = "Dit document gaat over landbouwbeleid en voedselveiligheid."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
predicted_labels = torch.sigmoid(outputs.logits) > 0.5  # Threshold for binary predictions
print(predicted_labels)
