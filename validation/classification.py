import json
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the dataset
dataset = load_dataset('coastalcph/multi_eurlex', 'nl')

# Load EUROVOC descriptors if needed
with open('./eurovoc_descriptors.json', 'r', encoding='utf-8') as json_file:
    eurovoc_concepts = json.load(json_file)

# Prepare the dataset for training
# Assuming you have train, validation, and test splits
train_dataset = dataset['train']
dev_dataset = dataset['validation']
test_dataset = dataset['test']

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


# Tokenize the datasets
train_encodings = train_dataset.map(tokenize_function, batched=True)
dev_encodings = dev_dataset.map(tokenize_function, batched=True)
test_encodings = test_dataset.map(tokenize_function, batched=True)

# Convert labels to tensor (assuming they are in numerical form already)
train_labels = torch.tensor(train_encodings['labels'])
dev_labels = torch.tensor(dev_encodings['labels'])
test_labels = torch.tensor(test_encodings['labels'])

# Create a custom dataset class for PyTorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets for training, validation, and testing
train_dataset = CustomDataset(train_encodings, train_labels)
dev_dataset = CustomDataset(dev_encodings, dev_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Load pre-trained BERT model for sequence classification
num_labels = len(set(train_labels.numpy()))  # Get number of unique labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    evaluation_strategy="epoch"  # Evaluate at the end of each epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,  # Use validation set for evaluation
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('my_finetuned_model')

# Evaluate the model on the test dataset
test_results = trainer.evaluate(test_dataset)
print(test_results)
