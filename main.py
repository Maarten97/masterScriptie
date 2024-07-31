import bwbXMLprocess
import rsCSVcopy
import rsClean
import versionControl
from bertprep import csvToTxt

dir_input_bwb = 'C:/Users/looijengam/Documents/Final/inputbwb'
dir_input_recht = 'C:/Users/looijengam/Documents/Final/inputrecht'
dir_output_bwb = 'C:/Users/looijengam/Documents/Final/bwb.csv'
dir_output_recht = 'C:/Users/looijengam/Documents/Final/recht.csv'
dir_output_txt = 'C:/Users/looijengam/Documents/Final/dataset.txt'
dir_output_combine = 'C:/Users/looijengam/Documents/Final/dataset.csv'


def create_bwb_csv():
    # Copy all files to one folder
    output_temp = dir_input_bwb + "/full"
    versionControl.folder_lookup(root_dir=dir_input_bwb,output_dir=output_temp)
    # Cleaning data and writing to CSV
    bwbXMLprocess.main(rootdir=output_temp, outputdir=dir_output_bwb)


def create_recht_csv():
    # Copy all files to one folder
    output_temp = dir_input_recht + "/full"
    rsCSVcopy.main(dir_input_recht, output_temp)
    # Cleaning data and writing to CSV
    rsClean.main(output_temp, dir_output_recht)


def combine_to_txt():
    listofdir = [dir_output_bwb, dir_output_recht]
    csvToTxt.main(inputs=listofdir, outputs=dir_output_txt, tempdir=dir_output_combine)


if __name__ == "__main__":
    create_bwb_csv()
