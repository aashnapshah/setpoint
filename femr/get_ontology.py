import sys
import pickle
from pathlib import Path

sys.path.append('../femr/')
from ontology import Ontology

# Configuration
ATHENA_PATH = "../data/athena_omop_vocabulary/"
SAVE_DIR = Path(".")

def load_ontology(path):
    """Load ontology from Athena database."""
    print(f"Loading ontology from {path}...")
    ontology = Ontology(path)
    print("Ontology loaded successfully")
    return ontology

def save_ontology(ontology, filename):
    """Save ontology to pickle file."""
    save_path = SAVE_DIR / filename
    print(f"Saving ontology to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(ontology, f)
    print("Ontology saved successfully")

# Load and save full ontology
ontology = load_ontology(ATHENA_PATH)
save_ontology(ontology, 'ontology.pkl')
