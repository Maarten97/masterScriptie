import os
import pandas as pd

directory_path = 'M:/BIT/Rechtspraak'


def count_rows_in_metadata_files(directory):
    total_rows = 0
    non_empty_full_text_rows = 0

    for filename in os.listdir(directory):
        if filename.endswith('metadata.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # Count total rows
            total_rows += len(df)

            # Count rows where 'full_text' is not empty
            non_empty_full_text_rows += df['full_text'].notna().sum()

    return total_rows, non_empty_full_text_rows


def multiple_years(dirpath):
    for folder in os.listdir(dirpath):
        year = os.path.basename(folder)[-4:]
        num_total, num_text = count_rows_in_metadata_files(os.path.join(dirpath, folder))
        print(f"Total rows in metadata files for year {year}: {num_total}")
        print(f"Rows with non-empty 'full_text' for year {year}: {num_text}")


if __name__ == '__main__':
    multiple_years(directory_path)
