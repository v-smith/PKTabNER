from drug_named_entity_recognition import find_drugs
import spacy


def find_drugnames(text: str) -> str | None:
    """
    Simple rule-based/static dictionary based drug NER.
    Find drug name spans in the input text and return a space-separated string of matches.
    Returns None if no drugs are found.
    """
    tokens = text.split()
    results = find_drugs(tokens)

    if not results:
        return None

    mentions = [res[0]["matching_string"] for res in results if "matching_string" in res[0]]
    return " ".join(sorted(set(mentions), key=mentions.index))

nlp = spacy.load("en_ner_bc5cdr_md")

def extract_drugs_spacy(text):
    """ spacy drug name extraction"""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]

