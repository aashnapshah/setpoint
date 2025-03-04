from __future__ import annotations

import collections
import functools
import os
import datasets
from typing import Any, Dict, Iterable, Optional, Set
import meds
import polars as pl
import pandas as pd

import hf_utils
print('Finished importing')
N_ROWS = 1000000

def _get_all_codes_map(batch) -> Set[str]:
    result = set()
    for events in batch["events"]:
        for event in events:
            for measurement in event["measurements"]:
                result.add(measurement["code"])
    return result


def _get_all_codes_agg(first: Set[str], second: Set[str]) -> Set[str]:
    first |= second
    return first


def clean_csv(filepath):
    """Clean problematic characters from CSV file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Remove problematic quotes and characters
        cleaned = line.replace('""', '"').replace('\\"', '"')
        cleaned_lines.append(cleaned)
    
    clean_filepath = filepath + '.clean'
    with open(clean_filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    return clean_filepath


class Ontology:
    def __init__(self, athena_path: str, code_metadata: meds.CodeMetadata = {}):
        """Create an Ontology from an Athena download and an optional meds Code Metadata structure.

        NOTE: This is an expensive operation.
        It is recommended to create an ontology once and then save/load it as necessary.
        """
        self.description_map: Dict[str, str] = {}
        self.parents_map: Dict[str, Set[str]] = collections.defaultdict(set)

        # Check file formats
        print("\nChecking CONCEPT.csv format...")
        with open(os.path.join(athena_path, "CONCEPT.csv"), 'r', encoding='utf8') as f:
            print("First 3 lines of CONCEPT.csv:")
            for i, line in enumerate(f):
                if i < 3:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break
                    
        print("\nChecking CONCEPT_RELATIONSHIP.csv format...")
        with open(os.path.join(athena_path, "CONCEPT_RELATIONSHIP.csv"), 'r', encoding='utf8') as f:
            print("First 3 lines of CONCEPT_RELATIONSHIP.csv:")
            for i, line in enumerate(f):
                if i < 3:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break
                    
        print("\nChecking CONCEPT_ANCESTOR.csv format...")
        with open(os.path.join(athena_path, "CONCEPT_ANCESTOR.csv"), 'r', encoding='utf8') as f:
            print("First 3 lines of CONCEPT_ANCESTOR.csv:")
            for i, line in enumerate(f):
                if i < 3:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break

        # Load from the athena path using pandas first
        print("\nReading CONCEPT.csv...")
        df = pd.read_csv(
            os.path.join(athena_path, "CONCEPT.csv"),
            sep="\t",
            #nrows=N_ROWS,
            dtype={
                'concept_name': str,
                'concept_id': int,
                'vocabulary_id': str,
                'concept_code': str,
                'standard_concept': str
            }
        )
        print(f"Successfully read {len(df)} rows with pandas")
        
        # Convert to polars and process
        concept = pl.from_pandas(df)
        processed_concepts = (
            concept.with_columns([
                (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code"),
                pl.col("concept_id").cast(pl.Int64),
                pl.col("concept_name"),
                pl.col("standard_concept").is_null()
            ])
            .select(["code", "concept_id", "concept_name", "standard_concept"])
            .rows()
        )
        print(f"Processed {len(processed_concepts)} concepts")

        # Build maps
        concept_id_to_code_map = {}
        non_standard_concepts = set()

        for code, concept_id, description, is_non_standard in processed_concepts:
            concept_id_to_code_map[concept_id] = code
            if code not in self.description_map:
                self.description_map[code] = description
            if is_non_standard:
                non_standard_concepts.add(concept_id)

        # Process relationships
        print('Reading CONCEPT_RELATIONSHIP.csv...')
        df = pd.read_csv(
            os.path.join(athena_path, "CONCEPT_RELATIONSHIP.csv"),
            sep="\t",
           # nrows=N_ROWS,
            dtype={
                'concept_id_1': int,
                'concept_id_2': int,
                'relationship_id': str
            }
        )
        relationship = pl.from_pandas(df)
        
        # Process ancestors
        print('Reading CONCEPT_ANCESTOR.csv...')
        df = pd.read_csv(
            os.path.join(athena_path, "CONCEPT_ANCESTOR.csv"),
            sep="\t",
            #nrows=N_ROWS,
            dtype={
                'descendant_concept_id': int,
                'ancestor_concept_id': int,
                'min_levels_of_separation': int
            }
        )
        ancestor = pl.from_pandas(df)

        # Filter relationships
        print('Filtering relationship')
        relationship = relationship.filter(
            (pl.col("relationship_id") == "Maps to") & 
            (pl.col("concept_id_1") != pl.col("concept_id_2"))
        )
        
        # Process relationships
        for row in relationship.select([
            pl.col("concept_id_1").cast(pl.Int64),
            pl.col("concept_id_2").cast(pl.Int64)
        ]).iter_rows():
            concept_id_1, concept_id_2 = row
            if concept_id_1 in non_standard_concepts and concept_id_1 in concept_id_to_code_map and concept_id_2 in concept_id_to_code_map:
                self.parents_map[concept_id_to_code_map[concept_id_1]].add(concept_id_to_code_map[concept_id_2])

        # Filter ancestor relationships
        ancestor = ancestor.filter(
            pl.col("min_levels_of_separation").cast(pl.Int64) == 1
        )
        
        # Process ancestor relationships
        for row in ancestor.select([
            pl.col("descendant_concept_id").cast(pl.Int64),
            pl.col("ancestor_concept_id").cast(pl.Int64)
        ]).iter_rows():
            concept_id, parent_concept_id = row  # Now row will be a tuple
            if concept_id in concept_id_to_code_map and parent_concept_id in concept_id_to_code_map:
                self.parents_map[concept_id_to_code_map[concept_id]].add(concept_id_to_code_map[parent_concept_id])

        # Have to add after OMOP to overwrite ...
        for code, code_info in code_metadata.items():
            if code_info.get("description") is not None:
                self.description_map[code] = code_info["description"]
            if code_info.get("parent_codes") is not None:
                self.parents_map[code] = set(code_info["parent_codes"])

        self.children_map = collections.defaultdict(set)
        for code, parents in self.parents_map.items():
            for parent in parents:
                self.children_map[parent].add(code)

        self.all_parents_map: Dict[str, Set[str]] = {}
        self.all_children_map: Dict[str, Set[str]] = {}

    def prune_to_dataset(
        self,
        dataset: datasets.Dataset,
        num_proc: int = 1,
        prune_all_descriptions: bool = False,
        remove_ontologies: Set[str] = set(),
    ) -> None:
        valid_codes = hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(_get_all_codes_map),
            _get_all_codes_agg,
            num_proc=num_proc,
            batch_size=1_000,
        )

        if prune_all_descriptions:
            self.description_map = {}

        all_parents = set()

        for code in valid_codes:
            all_parents |= self.get_all_parents(code)

        def is_valid(code):
            ontology = code.split("/")[0]
            return (code in valid_codes) or ((ontology not in remove_ontologies) and (code in all_parents))

        codes = self.children_map.keys() | self.parents_map.keys() | self.description_map.keys()
        for code in codes:
            m: Any
            if is_valid(code):
                for m in (self.children_map, self.parents_map):
                    m[code] = {a for a in m[code] if is_valid(a)}
            else:
                for m in (self.children_map, self.parents_map, self.description_map):
                    if code in m:
                        del m[code]

        self.all_parents_map = {}
        self.all_children_map = {}

        # Prime the pump
        for code in self.children_map.keys() | self.parents_map.keys():
            self.get_all_parents(code)

    def get_description(self, code: str) -> Optional[str]:
        """Get a description of a code."""
        return self.description_map.get(code)

    def get_children(self, code: str) -> Iterable[str]:
        """Get the children for a given code."""
        return self.children_map.get(code, set())

    def get_parents(self, code: str) -> Iterable[str]:
        """Get the parents for a given code."""
        return self.parents_map.get(code, set())

    def get_all_children(self, code: str) -> Set[str]:
        """Get all children, including through the ontology."""
        if code not in self.all_children_map:
            result = {code}
            for child in self.children_map.get(code, set()):
                result |= self.get_all_children(child)
            self.all_children_map[code] = result
        return self.all_children_map[code]

    def get_all_parents(self, code: str) -> Set[str]:
        """Get all parents, including through the ontology."""
        if code not in self.all_parents_map:
            result = {code}
            for parent in self.parents_map.get(code, set()):
                result |= self.get_all_parents(parent)
            self.all_parents_map[code] = result

        return self.all_parents_map[code]