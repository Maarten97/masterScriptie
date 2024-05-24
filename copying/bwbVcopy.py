import os
import xml.etree.ElementTree as ET
from lxml import etree


import bwbXMLprocess
import bwbWriteGeneral

# root_dir = 'D:/BWB/Origineel/202210_BWB_4'
root_dir = 'C:/Users/looij/Documents/BWB/Output'

namespaces = {
    "xml": "http://www.w3.org/XML/1998/namespace",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}


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
                bwbid = os.path.basename(direc)
                tree = ET.parse(filepath)
                root = tree.getroot()

                verdrag_elements = root.findall('.//verdrag[@xml:lang="nl"]', namespaces=namespaces)
                if verdrag_elements is not None:
                    for verdrag in verdrag_elements:
                        new_root = ET.ElementTree(verdrag).getroot()
                        text_elements_in_verdrag = new_root.findall('.//al')
                        for text_element in text_elements_in_verdrag:
                            al_text = bwbXMLprocess.process_xml_text(text_element)
                            if al_text is not None and al_text != "SKIP":
                                bwbWriteGeneral.write_general(direc, bwbid, al_text)
                else:
                    bwbWriteGeneral.write_error("bwbVcopy", "No Dutch verdrag in XML file: " + filename)


if __name__ == "__main__":
    dirlist = folder_lookup()
    language_check(dirlist)
    print("Code statusCheck executed")