import os
import xml.etree.ElementTree as ET

import bwbXMLprocess
import writeGeneral

root_dir = 'C:/Users/looij/Documents/BWB/Output'


def folder_lookup():
    dirlist = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBW'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlist.append(os.path.join(root_dir, bwbid))
    return dirlist


def to_csv(dlist):
    for direc in dlist:
        for filename in os.listdir(direc):
            if filename.endswith(".xml"):
                print("Currently loading XML file: " + filename)
                filepath = os.path.join(direc, filename)
                bwbid = os.path.basename(direc)
                tree = ET.parse(filepath)
                root = tree.getroot()

                for item in root.findall(".//al"):
                    al_text = bwbXMLprocess.process_xml_text(item)
                    if al_text is not None and al_text != "SKIP":
                        writeGeneral.write_general(direc, bwbid, al_text)


if __name__ == "__main__":
    dirlist = folder_lookup()
    to_csv(dirlist)
    print("Code bwbWcopy executed")
