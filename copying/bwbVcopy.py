import os
import xml.etree.ElementTree as ET

import bwbXMLprocess
import bwbWriteGeneral

# root_dir = 'D:/BWB/Origineel/202210_BWB_4'
root_dir = 'C:/Users/looij/Documents/BWB/Output'


# Requires already that are not more versions in the file
def folder_lookup():
    dirlist = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBV'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlist.append(os.path.join(root_dir, bwbid))
    return dirlist


def language_check(dlist):
    for direc in dlist:
        for filename in os.listdir(direc):
            if filename.endswith(".xml"):
                print("Currently loading XML file: " + filename)
                filepath = os.path.join(direc, filename)
                tree = ET.parse(filepath)
                root = tree.getroot()

                for verdrag in root.findall('.//verdrag[@xml:lang="nl"]'):
                    new_root = ET.ElementTree(verdrag).getroot()
                    text_elements_in_verdrag = new_root.findall('.//al')
                    for text_element in text_elements_in_verdrag:
                        # Write to CSV
                        al_text = bwbXMLprocess.process_xml_text(text_element)
                        if al_text is not None or al_text != "SKIP":
                            bwbWriteGeneral.write_general(direc, filename, al_text)


if __name__ == "__main__":
    dirlist = folder_lookup()
    language_check(dirlist)
    print("Code statusCheck executed")