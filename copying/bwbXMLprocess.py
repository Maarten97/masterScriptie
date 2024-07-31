import os
import re
import xml.etree.ElementTree as ET
import writeGeneral

invalid_stings = [
    "bevat overgangsrecht m.b.t. deze wijziging",
    "bevatten overgangsrecht m.b.t. deze wijziging",
    "Tekstplaatsing met vernummering",
    "De gegevens van inwerkingtreding zijn ontleend aan de bron van aankondiging van de tekstplaatsing",
    "De datum van inwerkingtreding is ontleend aan de bron van aankondiging van de tekstplaatsing",
    "Abusievelijk is een wijzigingsopdracht geformuleerd die niet geheel juist is",
    "Vervalt behoudens voor zover het betreft de toepassing of overeenkomstige toepassing van deze artikelen"
]

namespaces = {
    "xml": "http://www.w3.org/XML/1998/namespace",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

def folder_lookup(root_dir):
    dirlista = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBV') or bwbid.startswith('BWBR') or bwbid.startswith('BWBW'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlista.append(os.path.join(root_dir, bwbid))
    return dirlista


def get_text(dlist, output_dir):
    for direc in dlist:
        for filename in os.listdir(direc):

            if filename.endswith(".xml") and "manifest" not in filename:
                print("Currently loading XML file: " + filename)
                filepath = os.path.join(direc, filename)
                bwbid = os.path.basename(direc)

                try:
                    with open(filepath, "r", encoding="UTF-8") as file:
                        # tree = ET.parse(file)
                        xml_content = file.read()
                        xml_content = xml_content.encode('utf-8', 'replace').decode('utf-8')
                        tree = ET.ElementTree(ET.fromstring(xml_content))

                except (ET.ParseError, IOError) as e:
                    writeGeneral.write_error("XMLProcess", f"Error parsing XML file {filename}: {e}")
                    continue

                root = tree.getroot()
                total_text = ""

                if filename.startswith("BWBV"):
                    root = root.findall('.//verdrag[@xml:lang="nl"]', namespaces=namespaces)

                if root is not None:
                    for verdrag in root:
                        new_root = ET.ElementTree(verdrag).getroot()
                        text_elements_in_verdrag = new_root.findall('.//al')
                        for text_element in text_elements_in_verdrag:
                            al_text = process_xml_text(text_element)
                            if al_text and al_text != "SKIP":
                                total_text += al_text + " "
                else:
                    writeGeneral.write_error("XMLProcess", "No Dutch verdrag in XML file: " + filename)
                writeGeneral.write_general(output_dir, bwbid, total_text)


def process_xml_text(item):
    al_text = ''.join(item.itertext())

    # Removes all breaklines
    if "\n" in al_text:
        al_text = al_text.replace('\n', '')
    if "\r" in al_text:
        al_text = al_text.replace('\r', '')

    # Remove all extra whitespaces at the beginning or the end of the String
    al_text = al_text.strip()

    # Remove multiple whitespaces
    al_text = ' '.join(al_text.split())

    # Replace semicolons with dots
    al_text = al_text.replace(';', '.')
    al_text = al_text.replace(':', '.')
    al_text = al_text.replace('!', '.')
    al_text = al_text.replace('"', '')

    # Remove the ' " ' at the beginning or ending of every String
    if al_text.startswith('"'):
        al_text = al_text[1:]
    if al_text.endswith('"'):
        al_text = al_text[:-1]

    # Check if String is staring with '['
    if al_text.startswith('[') and al_text.endswith(']'):
        al_text = "SKIP"

    # Remove special characters except common punctuation
    al_text = re.sub(r'[!‘"@#$%^&*()_+=\[\]{};:\\|<>/?~`]', '', al_text)

    # Check if String is invalid according to definition
    for i in invalid_stings:
        if i in al_text:
            al_text = "SKIP"

    if al_text == "Vervallen" or al_text == "Vervallen." or al_text == "Besluit." or al_text == "Besluiten.":
        al_text = "SKIP"

    if al_text != "SKIP":
        newline = []
        words = al_text.split()
        for word in words:
            if word.isupper():
                word = word.lower()
            if len(word) == 1:
                continue
            if word == ',.':
                continue
            if len(re.findall(r'\d', word)) > 2:
                if word.endswith('.'):
                    word = '.'
                else:
                    continue
            if 'art.' in word.lower() or 'nr.' in word.lower():
                continue
            if word.startswith("'"):
                if not (len(word) > 2 and word[1] == "s"):
                    word = word[1:]
            if word.endswith("'"):
                word = word[:-1]
            if word == '.' and newline:
                newline[-1] += "."
            else:
                newline.append(word)
        al_text = ' '.join(newline)
    return al_text

# Method to call from outside class
def main(rootdir, outputdir):
    dirlists = folder_lookup(root_dir=rootdir)
    get_text(dirlists, output_dir=outputdir)


if __name__ == '__main__':
    root = 'C:/Users/looijengam/Documents/Output/Output/VersionControl'
    output = 'C:/Users/looijengam/Documents/Output/dataset.csv'

    dirlist = folder_lookup(root_dir=root)
    get_text(dirlist, output_dir=output)
    print("Code XMLProcess executed")

