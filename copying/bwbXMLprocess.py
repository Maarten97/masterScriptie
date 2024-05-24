
invalid_stings = [
    "bevat overgangsrecht m.b.t. deze wijziging",
    "bevatten overgangsrecht m.b.t. deze wijziging",
    "Tekstplaatsing met vernummering",
    "De gegevens van inwerkingtreding zijn ontleend aan de bron van aankondiging van de tekstplaatsing",
    "De datum van inwerkingtreding is ontleend aan de bron van aankondiging van de tekstplaatsing",
    "Abusievelijk is een wijzigingsopdracht geformuleerd die niet geheel juist is",
    "Vervalt behoudens voor zover het betreft de toepassing of overeenkomstige toepassing van deze artikelen "
]


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

    # Remove the ' " ' at the beginning or ending of every String
    if al_text.startswith('"'):
        al_text = al_text[1:]
    if al_text.endswith('"'):
        al_text = al_text[:-1]

    # Check if String is staring with '['
    if al_text.startswith('[') and al_text.endswith(']'):
        al_text = "SKIP"

    # Check if String is invalid according to definition
    for i in invalid_stings:
        if i in al_text:
            al_text = "SKIP"

    if al_text == "Vervallen" or al_text == "Vervallen.":
        al_text = "SKIP"

    return al_text
