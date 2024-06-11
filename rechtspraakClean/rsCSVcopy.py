import os
import pandas as pd
import writeGeneral

directory = 'PATH'
out_dir = 'PATH'


def process_csv(dirpath):
    for folder in os.listdir(dirpath):
        for filename in os.listdir(directory):
            if filename.endswith('metadata.csv'):
                file_path = os.path.join(directory, folder, filename)
                df = pd.read_csv(file_path)

                filtered_df = df[df['full_text'].notna()][['ID', 'full_text']]
                for _, row in filtered_df.iterrows():
                    writeGeneral.write_general(out_dir, str(row['ID']), row['full_text'])


if __name__ == '__main__':
    process_csv(directory)
