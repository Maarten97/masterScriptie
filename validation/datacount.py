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
level_2_ids = set(eurovoc_concepts["level_2"])
level_3_ids = set(eurovoc_concepts["level_3"])

# Get feature map info
classlabel = dataset.features["labels"].feature
def count_level_ids(sample):
    level_counts = {"level_1": 0, "level_2": 0, "level_3": 0}
    for label_id in sample['labels']:
        eurovoc_id = classlabel.int2str(label_id)

        # Increment count based on which level the Eurovoc ID belongs to
        if eurovoc_id in level_1_ids:
            level_counts["level_1"] += 1
        elif eurovoc_id in level_2_ids:
            level_counts["level_2"] += 1
        elif eurovoc_id in level_3_ids:
            level_counts["level_3"] += 1
    return level_counts


level_1_counts = defaultdict(int)
non_level_1_count = 0

# Retrieve IDs and descriptors from dataset
for sample in dataset:
    # print(f'DOCUMENT: {sample["celex_id"]}')
    # counts = count_level_ids(sample)
    # print(f'DOCUMENT: {sample["celex_id"]} | Level 1: {counts["level_1"]}, Level 2: {counts["level_2"]}, '
    #       f'Level 3: {counts["level_3"]}')


    # # DOCUMENT: 32006D0213
    # for label_id in sample['labels']:
    #     eurovoc_id = classlabel.int2str(label_id)
    #     if eurovoc_id in level_1_ids:
    #         eurovoc_desc = eurovoc_descriptors.get(eurovoc_id, {}).get('nl', 'Description not found')
    #         print(f'LABEL: id:{label_id}, eurovoc_id: {classlabel.int2str(label_id)}, \
    #             eurovoc_desc: {eurovoc_desc}')

            # LABEL: id: 1, eurovoc_id: '100160', eurovoc_desc: 'industry'

    for label_id in sample['labels']:
        eurovoc_id = classlabel.int2str(label_id)

        # Count occurrences of level 1 and non-level 1 labels
        if eurovoc_id in level_1_ids:
            level_1_counts[eurovoc_id] += 1
        else:
            non_level_1_count += 1

    # 7. Display results with Dutch description
print("Counts for each level 1 label with Dutch descriptions:")
for eurovoc_id, count in level_1_counts.items():
    eurovoc_desc = eurovoc_descriptors.get(eurovoc_id, {}).get('nl', 'Description not found')
    print(f'Eurovoc ID: {eurovoc_id}, Count: {count}, Description (NL): {eurovoc_desc}')

print(f"\nCount of all labels not in level 1: {non_level_1_count}")