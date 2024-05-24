import os
import xml.etree.ElementTree as ET

import bwbXMLprocess
import bwbWriteGeneral

root_dir = 'C:/Users/looij/Documents/BWB/Output'

namespaces = {
    "xml": "http://www.w3.org/XML/1998/namespace",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

def folder_lookup():
    dirlist = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBW'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlist.append(os.path.join(root_dir, bwbid))
    return dirlist

# TODO
def language_check(dirlist):
    pass


if __name__ == "__main__":
    dirlist = folder_lookup()
    language_check(dirlist)
    print("Code bwbWcopy executed")
