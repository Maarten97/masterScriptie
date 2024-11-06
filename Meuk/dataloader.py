# import json
# import csv
# from datasets import load_dataset
#
# OUTPUT_CSV = './eurovoc_descriptors_output.csv'
#
#
# def inspect_dataset_with_descriptors(dataseta, descriptor_file_path):
#     """
#     Displays basic information, samples from the dataset, and EUROVOC descriptors for each label.
#     """
#     # Load EUROVOC concept descriptors
#     with open(descriptor_file_path) as jsonl_file:
#         eurovoc_concepts = json.load(jsonl_file)
#
#     # Get feature information for the 'labels' feature in the dataset
#     classlabel = dataseta.features["labels"].feature
#
#     # Display dataset structure information
#     print("Dataset structure:")
#     print(f"Number of examples: {len(dataseta)}")
#     print(f"Features: {dataseta.features}")
#
#     # Display a few sample entries with EUROVOC descriptors
#     print("\nSample entries with descriptors:")
#     for i, sample in enumerate(dataseta):
#         print(f"\nDOCUMENT {i + 1}: {sample['celex_id']}")
#         for label_id in sample['labels']:
#             eurovoc_id = classlabel.int2str(label_id)
#             eurovoc_desc = eurovoc_concepts.get(eurovoc_id, {}).get('nl', 'Description not found')
#             print(f"  LABEL ID: {label_id}, EUROVOC ID: {eurovoc_id}, Description: {eurovoc_desc}")
#         if i >= 2:  # Display only a few samples for inspection
#             break
#
#
# def print_all_ids_with_descriptors(dataset, descriptor_file_path, output_file_path=OUTPUT_CSV):
#     """
#     Selects all IDs present in the dataset and writes the Label IDs, EUROVOC IDs, and descriptions to a CSV file.
#     """
#     # Load EUROVOC concept descriptors
#     with open(descriptor_file_path, 'r', encoding='utf-8') as jsonl_file:
#         eurovoc_concepts = json.load(jsonl_file)
#
#     # Get feature information for the 'labels' feature in the dataset
#     classlabel = dataset.features["labels"].feature
#
#     # Open the output CSV file for writing
#     with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile)
#
#         # Write header
#         csv_writer.writerow(['Label ID', 'EUROVOC ID', 'Description'])
#
#         # Iterate through the dataset and write Label IDs, EUROVOC IDs, and descriptions
#         for sample in dataset:
#             for label_id in sample['labels']:
#                 eurovoc_id = classlabel.int2str(label_id)
#                 # Get the Dutch description
#                 eurovoc_desc = eurovoc_concepts.get(eurovoc_id, {}).get('nl', 'Description not found')
#                 # Write a row to the CSV
#                 csv_writer.writerow([label_id, eurovoc_id, eurovoc_desc])
#
#
# def save_dataset():
    datasetsss = load_dataset('coastalcph/multi_eurlex', 'nl')
#     datasetsss.save_to_disk('./multi_eurlex_dataset')
#
#
# if __name__ == '__main__':
#     # Load the English part of the dataset, train split
#     datasetsss = load_dataset('coastalcph/multi_eurlex', 'nl', trust_remote_code=True)
#
#     # Inspect dataset and display EUROVOC descriptors
#     inspect_dataset_with_descriptors(datasetsss, './eurovoc_descriptors.json')
#
#     # Print all IDs with EUROVOC descriptors
#     # print_all_ids_with_descriptors(dataset, './eurovoc_descriptors.json')
#
#     # save_dataset()
