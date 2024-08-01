import os
import pandas as pd
import writeGeneral


def process_csv(dirpath, out_dir):
    for folder in os.listdir(dirpath):
        for filename in os.listdir(os.path.join(dirpath, folder)):
            if filename.endswith('metadata.csv'):
                file_path = os.path.join(dirpath, folder, filename)
                df = pd.read_csv(file_path)

                filtered_df = df[df['full_text'].notna()][['ecli', 'full_text']]
                writedir = os.path.join(out_dir + '/' + os.path.basename(folder)[-4:] + '.csv')
                for _, row in filtered_df.iterrows():
                    writeGeneral.write_general(writedir, str(row['ecli']), row['full_text'])


def main(dirs, out):
    process_csv(dirs, out)


if __name__ == '__main__':
    directory = 'C:/Programming/Dataset/Rechtspraak'
    out_dir = 'C:/Programming/Dataset/RechtspraakOutput'
    process_csv(directory, out_dir)
