from copying import bwbXMLprocess
from rechtspraakClean import rsCSVcopy
from rechtspraakClean import rsClean
from cleaning import versionControl
from bertprep import csvToTxt

dir_input_bwb = 'M:/BIT/BWB'
dir_temp_bwb = 'M:/BIT/BWB/full'
dir_input_recht = 'M:/BIT/Rechtspraak'
dir_output_bwb = 'M:/BIT/bwb.csv'
dir_output_recht = 'M:/BIT/recht.csv'
dir_output_txt = 'M:/BIT/dataset.txt'
dir_output_combine = 'M:/BIT/dataset.csv'


def create_bwb_csv():
    print('Copy all BWB files to one folder')
    versionControl.folder_lookup(root_dir=dir_input_bwb, output_dir=dir_temp_bwb)
    print('Cleaning BWB data and writing to CSV')
    bwbXMLprocess.main(rootdir=dir_temp_bwb, outputdir=dir_output_bwb)


def create_recht_csv():
    print('Copy all files to one folder')
    output_temp = dir_input_recht + "/full"
    rsCSVcopy.main(dir_input_recht, output_temp)
    print('Cleaning data and writing to CSV')
    rsClean.main(output_temp, dir_output_recht)


def combine_to_txt():
    print('Combine datasets and put in TXT file')
    listofdir = [dir_output_bwb, dir_output_recht]
    csvToTxt.main(inputs=listofdir, outputs=dir_output_txt, tempdir=dir_output_combine)


if __name__ == "__main__":
    create_bwb_csv()
    create_recht_csv()
    combine_to_txt()