import json
from datasets import load_dataset
from collections import defaultdict

# Load the English part of the dataset
dataset = load_dataset('coastalcph/multi_eurlex', 'nl', split='validation')

# Load (label_id, descriptor) mapping
with open('./eurovoc_descriptors.json') as jsonl_file:
    eurovoc_descriptors = json.load(jsonl_file)

# 3. Load the Eurovoc concepts to filter for level 1 IDs
with open('./eurovoc_concepts.json') as f:
    eurovoc_concepts = json.load(f)

# 4. Extract level 1 Eurovoc IDs
level_1_ids = set(eurovoc_concepts["level_1"])

# Get feature map info
classlabel = dataset.features["labels"].feature
print(classlabel)