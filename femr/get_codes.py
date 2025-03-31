import sys
import pickle
from pathlib import Path
from ontology import Ontology

ATHENA_PATH = "../data/athena_omop_vocabulary/"
SAVE_DIR = Path("../data/")
PICKLE_PATH = SAVE_DIR / 'ontology.pkl'

START_CODES = {
        "chronic_heart_disease": ["ICD9CM/I48", "ICD10CM/I48"],
        "chronic_kidney_disease": ["ICD9CM/585", "ICD10CM/N18"],            
        "type_2_diabetes": ["ICD9CM/250.x0", "ICD9CM/250.x2", "ICD10/E11", "SNOMED/44054006"],
        "MACE": ["ICD9CM/410", "ICD9CM/M/428", "ICD9CM/431", "ICD9CM/432", "ICD9CM/434", 
                 "ICD10CM/I21", "ICD10CM/I50", "ICD10CM/I61", "ICD10CM/I62", "ICD10CM/I63"],
        "MDS": ["ICD9CM/238.72–238.75", "ICD10CM/D46"],
        "osteoporosis": ["ICD9CM/733.0", "ICD9CM/733.1", "ICD10CM/M80", "ICD10CM/M81"],
    }

def load_ontology(path):
    """Load ontology from Athena database or pickle file."""
    if PICKLE_PATH.exists():
        print(f"Loading ontology from pickle file {PICKLE_PATH}...")
        with open(PICKLE_PATH, 'rb') as f:
            ontology = pickle.load(f)
        print("Ontology loaded successfully from pickle")
    else:
        print(f"Loading ontology from {path}...")
        ontology = Ontology(path)
        print("Ontology loaded successfully from Athena")
        
        # Save to pickle for future use
        save_ontology(ontology, 'ontology.pkl')
    return ontology

def save_ontology(ontology, filename):
    """Save ontology to pickle file."""
    save_path = SAVE_DIR / filename
    print(f"Saving ontology to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(ontology, f)
    print("Ontology saved successfully")
      
def get_icd_codes() -> dict:
    """Get ICD codes."""
    with open('icd_codes.pkl', 'rb') as f:
        icd_codes = pickle.load(f)
    return icd_codes

def expand_icd_codes(icd_codes: dict) -> dict:
    """Expand ICD codes dictionary by including related codes and their descriptions."""
    processed_codes = set()
    expanded_codes = {disease: {} for disease in icd_codes}

    for disease, codes in icd_codes.items():
        for code in codes:
            if code in processed_codes:
                continue
            
            processed_codes.add(code)
            # Add original code with its description
            expanded_codes[disease][code] = ontology.get_description(code)
            
            # Get and process parent codes
            parents = ontology.get_parents(code)
            for parent in parents or []:
                # Get and process children of parent codes
                children = ontology.get_children(parent)
                for child in children or []:
                    if child not in processed_codes and child not in expanded_codes[disease]:
                        expanded_codes[disease][child] = ontology.get_description(child)
                        processed_codes.add(child)
    
    return expanded_codes

def save_icd_codes(icd_codes: dict, filename: str = '../data/icd_codes.pkl') -> None:
    """Save ICD codes dictionary to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(icd_codes, f)
    print(f"ICD codes saved to {filename}")
    

def load_icd_codes(filename: str = '../data/icd_codes.pkl') -> dict:
    """Load ICD codes dictionary from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def remove_codes(icd_codes: dict) -> dict:
        # FILTER 
    chronic_heart_disease_remove = [
        "SNOMED/233891009",  # Sinoatrial node tachycardia
        "SNOMED/233893007",  # Re-entrant atrial tachycardia
        "SNOMED/195100002",  # Sinoatrial node dysfunction NOS
    ]
    # remove codes from the icd_codes dictionary
    disease = 'chronic_heart_disease'
    for code in chronic_heart_disease_remove:
        if code in icd_codes[disease]:
            del icd_codes[disease][code]

    type_2_diabetes_remove = [
        "SNOMED/46635009",  # Type 1 diabetes mellitus
        "SNOMED/190418009",  # Diabetes mellitus, juvenile type, with other specified manifestation
        "SNOMED/532401000000104",  # Diabetes mellitus, juvenile type, with no mention of complication
        "SNOMED/267379000",  # Diabetes mellitus, juvenile type, with no mention of complication
        "SNOMED/190322003",  # Diabetes mellitus: [juvenile type, with no mention of complication] or [insulin dependent]
        "SNOMED/703136005",  # Diabetes mellitus in remission
        "SNOMED/8801005",  # Secondary diabetes mellitus
        "SNOMED/890171006",  # Ketosis-prone diabetes mellitus
        "SNOMED/609569007",  # Diabetes mellitus due to genetic defect in insulin action
        "SNOMED/609568004",  # Diabetes mellitus due to genetic defect in beta cell function
        "SNOMED/105401000119101",  # Diabetes mellitus due to pancreatic injury
        "SNOMED/122451000000107",  # Insulin autoimmune syndrome without complication
        "SNOMED/112971000000100",  # Insulin autoimmune syndrome without complication
        "SNOMED/335621000000101",  # Maternally inherited diabetes mellitus
        "SNOMED/722206009",  # Pancreatic hypoplasia, diabetes mellitus, congenital heart disease syndrome
        "SNOMED/123763000",  # Houssay's syndrome
        "SNOMED/733072002"  # Stimmler syndrome
    ]
    disease = 'type_2_diabetes'
    for code in type_2_diabetes_remove:
        if code in icd_codes[disease]:
            del icd_codes[disease][code]

    non_mace_codes = [
        # Codes that do not map to MACE (Major Adverse Cardiovascular Events)
        "SNOMED/140881000119109",  # Spontaneous cerebral hemorrhage with compression of brain
        "ICD10CM/S06.36",  # Traumatic hemorrhage of cerebrum, unspecified
        "SNOMED/288276001",  # Fetal cerebral hemorrhage
        "SNOMED/45639009",  # Hereditary cerebral amyloid angiopathy, Icelandic type
        "SNOMED/450418003",  # Cerebral hemorrhage due to trauma
        "SNOMED/32728005",  # Hemorrhage due to ruptured congenital cerebral aneurysm
        "SNOMED/237702003",  # Pituitary hemorrhage
        "SNOMED/1263990007",  # Fetal nontraumatic intracranial hemorrhage
        "SNOMED/1258879002",  # Intracranial hemorrhage following administration of thrombolytic agent
        "SNOMED/450410005",  # Intracranial hemorrhage following injury
        "ICD10CM/P52.8",  # Other intracranial (nontraumatic) hemorrhages of newborn
        "SNOMED/111615001",  # Open skull fracture with intracranial hemorrhage
        "ICD10CM/P52.6",  # Cerebellar (nontraumatic) and posterior fossa hemorrhage of newborn
        "ICD10CM/P52.4",  # Intracerebral (nontraumatic) hemorrhage of newborn
        "SNOMED/723857007",  # Silent micro-hemorrhage of brain
        "SNOMED/450425005",  # Intracranial hematoma
        "SNOMED/35486000",  # Subdural hemorrhage
        "SNOMED/451035002",  # Subpial intracranial hemorrhage
        "SNOMED/82999001",  # Epidural intracranial hemorrhage
        "SNOMED/737160009",  # Dissection of basilar artery
        "SNOMED/723141009",  # Intracranial hemorrhage co-occurrent and due to complex wound of head
        "SNOMED/90099008",  # Subcortical leukoencephalopathy
        "SNOMED/95830009",  # Pituitary infarction
        "SNOMED/230523009",  # Infarction of optic radiation
        "ICD10CM/P91.823",  # Neonatal cerebral infarction, bilateral
        "SNOMED/302904002",  # Infarction of visual cortex
        "SNOMED/230518009",  # Infarction of optic tract
        "SNOMED/441526008",  # Infarct of cerebrum due to iatrogenic cerebrovascular accident
        "SNOMED/413102000",  # Infarction of basal ganglia
        "ICD10CM/P91.822",  # Neonatal cerebral infarction, left side of brain
        "SNOMED/1231168008",  # Malignant middle cerebral artery syndrome
        "SNOMED/276219001",  # Occipital cerebral infarction
        "SNOMED/230698000",  # Lacunar infarction
        "SNOMED/230693009"  # Anterior cerebral circulation infarction
    ]
    disease = 'MACE'
    for code in non_mace_codes:
        if code in icd_codes[disease]:
            del icd_codes[disease][code]

    non_osteoporosis_codes = [
        # Codes that do not map to Osteoporosis
        "ICD9CM/733.1",  # Pathologic fracture (not necessarily related to osteoporosis)
        "SNOMED/287070000",  # Pathological fracture - ankle and/or foot
        "ICD10CM/M84.459A",  # Pathological fracture, hip, unspecified, initial encounter for fracture
        "ICD10CM/M84.68",  # Pathological fracture in other disease, other site
        "ICD10CM/M84.48XG",  # Pathological fracture, other site, subsequent encounter for fracture with delayed healing
        "SNOMED/366908006",  # Pathological fracture due to metastatic bone disease
        "SNOMED/11315201000119108",  # Pathological fracture of mandible
        "SNOMED/658881000000100",  # Other specified pathological fracture
        "SNOMED/203463009",  # Other specified pathological fracture
        "ICD10CM/M84.68XA",  # Pathological fracture in other disease, other site, initial encounter for fracture
        "SNOMED/409667007",  # Pathological fracture of femur
        "SNOMED/426115002",  # Pathological fracture of pelvis
        "ICD10CM/M84.60XA",  # Pathological fracture in other disease, unspecified site, initial encounter for fracture
        "SNOMED/11315041000119101",  # Pathological fracture of right hip
        "SNOMED/11315081000119106",  # Pathological fracture of left hip
        "ICD10CM/M84.40XD",  # Pathological fracture, unspecified site, subsequent encounter for fracture with routine healing
        "SNOMED/240200008",  # Collapse of spine NOS
        "SNOMED/240199005",  # Collapse of vertebra NOS
        "SNOMED/287069001",  # Pathological fracture - lower leg
        "ICD10CM/M84.60",  # Pathological fracture in other disease, unspecified site
        "SNOMED/704169000",  # Pathological fracture of rib
    ]
    disease = 'osteoporosis'
    for code in non_osteoporosis_codes:
        if code in icd_codes[disease]:
            del icd_codes[disease][code]
    return icd_codes

# Example usage:
if __name__ == "__main__":
    # Create and expand ICD codes
    ontology = load_ontology(ATHENA_PATH)
    icd_codes = expand_icd_codes(START_CODES)
    icd_codes = remove_codes(icd_codes)
    save_icd_codes(icd_codes)
    loaded_codes = load_icd_codes()
    # print the number of codes for each disease in the icd_codes dictionary
    for disease in loaded_codes:
        print(f"{disease}: {len(loaded_codes[disease])}")


