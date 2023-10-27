import pandas as pd
import numpy as np
import fastobo
from tqdm import tqdm
import os
import re
import requests
import inflection
from datetime import timezone

from utils import Singleton, Database, get_best_match, camelize, SQLdatabase, profile

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "databases"
log = logging.getLogger(logging_name)


###     --- DATABASES ---       ###


# @profile()
class NCBI(Database, metaclass=Singleton):
    """
    NCBI reference class
    Contains info about genes
    For mapping all protein names and aliases to the common nomenclature of official symbols of the NCBI gene database
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Public Domain",
            license_url="https://www.ncbi.nlm.nih.gov/home/about/policies/#copyright",
        )
        self.__geneinfo = self._add_file(
            url="https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            update=update,
        )
        self.__geneinfo.content = self.__geneinfo.content.query(
            "Full_name_from_nomenclature_authority != '-'"
        ).drop_duplicates(
            subset="Symbol_from_nomenclature_authority", keep=False, ignore_index=True,
        )  # keep only unique official symbols
        # NCBI official symbols
        self.__ncbi_symbols = set(
            self.__geneinfo()["Symbol_from_nomenclature_authority"]
        )
        # NCBI official ids
        self.__ncbi_ids = set(self.__geneinfo()["GeneID"].astype({"GeneID": int}))

        # NCBI unique synonyms (not ambiguous)
        self.__unique_synonyms = {}
        shared_synonyms = {}
        for symbol, synonyms, ncbi_symbol in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "Synonyms", "Symbol"]
        ].itertuples(index=False):
            if synonyms != "-":
                for synonym in synonyms.split("|") + [ncbi_symbol]:
                    # ambiguity check
                    if synonym in shared_synonyms:
                        shared_synonyms[synonym].append(symbol)
                    elif synonym in self.__unique_synonyms:
                        shared_synonyms[synonym] = [
                            self.__unique_synonyms[synonym],
                            symbol,
                        ]
                        del self.__unique_synonyms[synonym]
                    else:
                        self.__unique_synonyms[synonym] = symbol

        self.__gene2name = self.__geneinfo()[
            [
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
            ]
        ].rename(
            columns={
                "Symbol_from_nomenclature_authority": "geneSymbol",
                "Full_name_from_nomenclature_authority": "geneName",
            }
        )
        self.__gene2name_asdict = self.__gene2name.set_index("geneSymbol")[
            "geneName"
        ].to_dict()
        self.__id2symbol = (
            self.__geneinfo()[["GeneID", "Symbol_from_nomenclature_authority"]]
            .rename(
                columns={
                    "GeneID": "geneId",
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                }
            )
            .astype({"geneId": int})
        )
        self.__id2symbol_asdict = self.__id2symbol.set_index("geneId")[
            "geneSymbol"
        ].to_dict()

        self.__symbol2id = (
            self.__geneinfo()[["Symbol_from_nomenclature_authority", "GeneID"]]
            .rename(
                columns={
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                    "GeneID": "geneId",
                }
            )
            .astype({"geneId": int})
        )
        self.__symbol2id_asdict = self.__symbol2id.set_index("geneSymbol")[
            "geneId"
        ].to_dict()

        self.__omim2ncbi = set()
        self.__hgnc2ncbi = set()
        self.__ensembl2ncbi = set()
        for symbol, xref in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "dbXrefs"]
        ].itertuples(index=False):
            if xref != "-":
                for key, value in {
                    id.split(":", 1)[0]: id.split(":", 1)[1] for id in xref.split("|")
                }.items():
                    if key == "MIM":
                        self.__omim2ncbi.add((value, symbol))
                    elif key == "HGNC":
                        self.__hgnc2ncbi.add((value, symbol))
                    elif key == "Ensembl":
                        self.__ensembl2ncbi.add((value, symbol))
        self.__omim2ncbi = pd.DataFrame(self.__omim2ncbi, columns=["OMIM", "NCBI"])
        self.__omim2ncbi = self.__omim2ncbi[
            ~(
                self.__omim2ncbi["OMIM"].duplicated(keep=False)
                + self.__omim2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__omim2ncbi_asdict = self.__omim2ncbi.set_index("OMIM")["NCBI"].to_dict()
        self.__hgnc2ncbi = pd.DataFrame(self.__hgnc2ncbi, columns=["HGNC", "NCBI"])
        self.__hgnc2ncbi = self.__hgnc2ncbi[
            ~(
                self.__hgnc2ncbi["HGNC"].duplicated(keep=False)
                + self.__hgnc2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__hgnc2ncbi_asdict = self.__hgnc2ncbi.set_index("HGNC")["NCBI"].to_dict()
        self.__ensembl2ncbi = pd.DataFrame(
            self.__ensembl2ncbi, columns=["Ensembl", "NCBI"]
        )
        self.__ensembl2ncbi = self.__ensembl2ncbi[
            ~(
                self.__ensembl2ncbi["Ensembl"].duplicated(keep=False)
                + self.__ensembl2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__ensembl2ncbi_asdict = self.__ensembl2ncbi.set_index("Ensembl")[
            "NCBI"
        ].to_dict()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def symbols(self):
        """returns NCBI official symbols"""
        return self.__ncbi_symbols

    @property
    def database(self):
        return self.__geneinfo().copy()

    def check_symbol(self, symbol=None, aliases=[]):
        """
        Checks if symbols or aliases are ambiguous or can be mapped to unique NCBI symbol
        Returns the official NCBI symbol if there is one, False otherwise
        """
        if symbol is pd.NA or (not symbol and not aliases):
            return None
        elif symbol in self.__ncbi_symbols:
            return symbol
        elif symbol in self.__unique_synonyms:
            return self.__unique_synonyms[symbol]
        elif aliases:
            for alias in aliases:
                if alias in self.__ncbi_symbols:
                    return alias
                elif alias in self.__unique_synonyms:
                    return self.__unique_synonyms[alias]
        return None

    def get_symbol_by_id(self, id):
        """
        Checks if id in official NCBI ids
        Returns the official corresponding NCBI symbol if there is one, None otherwise
        """
        if not isinstance(id, int):
            if isinstance(id, str) and ("NCBI:" in id or "NCBIGene:" in id):
                id = id.split(":")[-1]
            try:
                id = int(id)
            except:
                return None
        if not id:
            return None
        elif id in self.__ncbi_ids:
            return self.check_symbol(self.__id2symbol_asdict.get(id))
        else:
            return None

    def get_id_by_symbol(self, symbol):
        """
        Checks if symbol in official NCBI symbols
        Returns the official corresponding NCBI id if there is one, None otherwise
        """
        if isinstance(symbol, str) and ("NCBI:" in symbol or "NCBIGene:" in symbol):
            symbol = symbol.split(":")[-1]
        if not symbol:
            return None
        elif symbol in self.__ncbi_symbols:
            return int(self.__symbol2id_asdict.get(self.check_symbol(symbol)))
        else:
            return None

    @property
    def id2symbol(self):
        return self.__id2symbol.copy()

    @property
    def symbol2id(self):
        return self.__symbol2id.copy()

    def get_name(self, symbol):
        """
        Given an official symbol returns the Full_name_from_nomenclature_authority
        """
        return self.__gene2name_asdict.get(symbol, f"{symbol} Not Found")

    @property
    def gene2name(self):
        return self.__gene2name.copy()

    @property
    def omim2ncbi(self):
        return self.__omim2ncbi.copy()

    @property
    def omim2ncbi_asdict(self):
        return self.__omim2ncbi_asdict.copy()

    def get_symbol_by_omim(self, omim):
        """
        Given a OMIM id returns the NCBI gene symbol
        """
        return self.__omim2ncbi_asdict.get(omim, f"{omim} Not Found")

    @property
    def hgnc2ncbi(self):
        return self.__hgnc2ncbi.copy()

    @property
    def hgnc2ncbi_asdict(self):
        return self.__hgnc2ncbi_asdict.copy()

    def get_symbol_by_hgnc(self, hgnc):
        """
        Given a HGNC id returns the NCBI gene symbol
        """
        return self.__hgnc2ncbi_asdict.get(hgnc, f"{hgnc} Not Found")

    @property
    def ensembl2ncbi(self):
        return self.__ensembl2ncbi.copy()

    @property
    def ensembl2ncbi_asdict(self):
        return self.__ensembl2ncbi_asdict.copy()

    def get_symbol_by_ensembl(self, ensembl):
        """
        Given a Ensembl id returns the NCBI gene symbol
        """
        return self.__ensembl2ncbi_asdict.get(ensembl, f"{ensembl} Not Found")


# @profile()
class HGNC(Database, metaclass=Singleton):
    """
    HGNC reference class
    Contains info about genes
    For mapping all protein names, aliases and symbols to the approved human gene nomenclature
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available",
            license_url="https://www.genenames.org/about/",
        )
        self.__db = self._add_file(
            url="http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_types/gene_with_protein_product.txt",
            update=update,
        )
        # HGNC official symbols
        self.__hgnc_symbols = tuple(self.__db()["symbol"])
        # HGNC official ids
        self.__hgnc_ids = tuple(self.__db()["hgnc_id"])

        # HGNC unique synonyms (not ambiguous)
        self.__unique_synonyms = {}
        shared_synonyms = {}
        for symbol, synonyms in (
            self.__db()[["symbol", "alias_symbol"]]
            .dropna(subset=["alias_symbol"])
            .itertuples(index=False)
        ):
            for synonym in synonyms.split("|"):
                # ambiguity check
                if synonym in shared_synonyms:
                    shared_synonyms[synonym].append(symbol)
                elif synonym in self.__unique_synonyms:
                    shared_synonyms[synonym] = [
                        self.__unique_synonyms[synonym],
                        symbol,
                    ]
                    del self.__unique_synonyms[synonym]
                else:
                    self.__unique_synonyms[synonym] = symbol

        self.__gene2name = self.__db()[["symbol", "name"]].rename(
            columns={"symbol": "geneSymbol", "name": "geneName",}
        )
        self.__gene2name_asdict = self.__gene2name.set_index("geneSymbol")[
            "geneName"
        ].to_dict()
        self.__id2symbol = self.__db()[["hgnc_id", "symbol"]].rename(
            columns={"hgnc_id": "geneId", "symbol": "geneSymbol"}
        )
        self.__id2symbol_asdict = self.__id2symbol.set_index("geneId")[
            "geneSymbol"
        ].to_dict()

        self.__symbol2id = self.__db()[["symbol", "hgnc_id"]].rename(
            columns={"symbol": "geneSymbol", "hgnc_id": "geneId"}
        )
        self.__symbol2id_asdict = self.__symbol2id.set_index("geneSymbol")[
            "geneId"
        ].to_dict()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def symbols(self):
        """returns HGNC official symbols"""
        return self.__hgnc_symbols

    @property
    def database(self):
        return self.__db().copy()

    def check_symbol(self, symbol, aliases=[]):
        """
        Checks if symbols or aliases are ambiguous or can be mapped to unique HGNC symbol
        Returns the official HGNC symbol if there is one, False otherwise
        """
        if not symbol:
            return None
        elif symbol in self.__hgnc_symbols:
            return symbol
        elif symbol in self.__unique_synonyms:
            return self.__unique_synonyms[symbol]
        else:
            for alias in aliases:
                if alias in self.__hgnc_symbols:
                    return alias
                elif alias in self.__unique_synonyms:
                    return self.__unique_synonyms[alias]
        return None

    def get_symbol_by_id(self, id):
        """
        Checks if id in official HGNC ids and returns the official corresponding HGNC symbol if there is one, None otherwise
        """
        if not id:
            return None
        if not isinstance(id, str):
            id = str(id)
        if not id.startswith("HGNC:"):
            id = "HGNC:" + id
        if not id:
            return None
        elif id in self.__hgnc_ids:
            return self.check_symbol(self.__id2symbol_asdict.get(id))
        else:
            return None

    def get_id_by_symbol(self, symbol):
        """
        Checks if symbol in official HGNC symbols
        Returns the official corresponding HGNC id if there is one, None otherwise
        """
        if not symbol:
            return None
        elif symbol in self.__hgnc_symbols:
            return self.__symbol2id_asdict.get(self.check_symbol(symbol))
        else:
            return None

    @property
    def id2symbol(self):
        return self.__id2symbol.copy()

    def get_name(self, symbol):
        """
        Given an official symbol returns the gene name
        """
        return self.__gene2name_asdict.get(symbol, f"{symbol} Not Found")

    @property
    def gene2name(self):
        return self.__gene2name.copy()


# @profile()
class UniProt(Database, metaclass=Singleton):
    """
    UniProt reference class
    Contains info about genes
    For converting UniProtKB entries to NCBI gene symbols
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://www.uniprot.org/help/license",
        )
        self.__uniprot2symbol = self._add_file(
            url="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz",
            update=update,
            names=["uniprotId", "type", "geneSymbol"],
            query="type == 'Gene_Name'",
            final_columns=["uniprotId", "geneSymbol"],
        )
        self.__uniprot2symbol_dict = {uniprot: symbol for uniprot, symbol, _ in self.__uniprot2symbol().itertuples(index=False)}

        log.info(f"{self.__class__.__name__} ready!")

    def get_symbol(self, uniprot_id):
        return self.__uniprot2symbol_dict.get(uniprot_id, None)

    @property
    def uniprot2symbol(self):
        return self.__uniprot2symbol().copy()


# @profile()
class DisGeNET(Database, metaclass=Singleton):
    """
    DisGeNET reference class
    Contains info about gene-disease associations
    """

    def __init__(self, update=None):
        try:
            email = os.environ["DISGENET_EMAIL"]
            password = os.environ["DISGENET_PASSWORD"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                email = credentials["DISGENET_EMAIL"]
                password = credentials["DISGENET_PASSWORD"]
            except:
                raise RuntimeError(
                    "No DisGeNET credentials found, "
                    "gene-disease associations will not be collected"
                )
        auth_url = "https://www.disgenet.org/api/auth/"
        response = requests.post(auth_url, data={"email": email, "password": password})
        token = response.json()["token"]
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC-SA 4.0",
            license_url="https://www.disgenet.org/legal",
            registration_required=True,
            session=session,
            requirements=[NCBI],
        )
        self.__disease_mapping = self._add_file(
            url="https://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz",
            usecols=["diseaseId", "vocabulary", "code"],
        )
        self.__umls2mondo = {
            (umls, f"MONDO:{mondo}")
            for umls, mondo in self.__disease_mapping()
            .query("vocabulary == 'MONDO'")[["diseaseId", "code"]]
            .itertuples(index=False)
        }
        self.__db = self._add_file(
            url="https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz",
            dtype={
                "geneId": int,
                "geneSymbol": "string",
                "DSI": float,
                "DPI": float,
                "diseaseId": "string",
                "diseaseName": "string",
                "diseaseType": "string",
                "diseaseClass": "string",
                "diseaseSemanticType": "string",
                "score": float,
                "EI": float,
                "YearInitial": "string",
                "YearFinal": "string",
                "NofPmids": int,
                "NofSnps": int,
                "source": "string",
            },
        )
        self.__db.content["geneSymbol"] = self.__db.content["geneSymbol"].map(
            self.__NCBI.check_symbol
        )
        self.__db.content.dropna(subset=["geneSymbol"], inplace=True)
        self.__db.content.drop_duplicates(ignore_index=True, inplace=True)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def umls2mondo(self):
        return self.__umls2mondo.copy()

    @property
    def database(self):
        return self.__db().copy()

    @property
    def disease_mapping(self):
        return self.__disease_mapping().copy()


# @profile()
class MONDO(Database, metaclass=Singleton):
    """
    MONDO Ontology by Monarch Initiative reference class
    Contains info about diseases
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://mondo.monarchinitiative.org/#license",
            requirements=[DisGeNET],
        )
        self.__db = self._add_file(
            url="http://purl.obolibrary.org/obo/mondo.obo",
            version_marker="data-version",
        )

        self.__umls2mondo = self.__DisGeNET.umls2mondo.copy()
        self.__mondo2name = set()
        self.__omim2mondo = {}
        self.__orpha2mondo = {}
        self.__doid2mondo = {}
        self.__mondoRelationship = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            is_obsolete = False
            relationship_tmp = []
            for clause in frame:
                if isinstance(clause, fastobo.term.IsObsoleteClause):
                    is_obsolete = True
                    break
                elif isinstance(clause, fastobo.term.XrefClause) and hasattr(
                    clause.xref.id, "prefix"
                ):
                    if clause.xref.id.prefix in ["UMLS", "UMLS_CUI"]:
                        self.__umls2mondo.add(
                            (clause.xref.id.local, f"MONDO:{frame.id.local}")
                        )
                    elif clause.xref.id.prefix == "OMIM":
                        self.__omim2mondo[
                            f"OMIM:{clause.xref.id.local}"
                        ] = f"MONDO:{frame.id.local}"
                    elif clause.xref.id.prefix == "Orphanet":
                        self.__orpha2mondo[
                            f"ORPHA:{clause.xref.id.local}"
                        ] = f"MONDO:{frame.id.local}"
                    elif clause.xref.id.prefix == "DOID":
                        self.__doid2mondo[
                            f"DOID:{clause.xref.id.local}"
                        ] = f"MONDO:{frame.id.local}"
                elif hasattr(frame.id, "prefix"):
                    if (
                        isinstance(clause, fastobo.term.RelationshipClause)
                        and clause.typedef != "never_in_taxon"
                        and clause.typedef != "excluded_subClassOf"
                    ):
                        relationship_tmp.append(
                            (str(frame.id), clause.typedef.escaped, str(clause.term))
                        )
                    elif isinstance(clause, fastobo.term.IsAClause):
                        relationship_tmp.append(
                            (str(frame.id), "is_a", str(clause.term))
                        )
            if not is_obsolete and isinstance(frame[0], fastobo.term.NameClause):
                self.__mondo2name.add((f"MONDO:{frame.id.local}", frame[0].name))
                self.__mondoRelationship.update(relationship_tmp)

        self.__ontology = pd.DataFrame(
            self.__mondoRelationship, columns=["subject", "relation", "object"]
        )
        self.__ontology = self.__ontology.merge(
            pd.DataFrame(self.__mondo2name, columns=["subject", "subjectName"]),
            on="subject",
        )
        self.__ontology = self.__ontology.merge(
            pd.DataFrame(self.__mondo2name, columns=["object", "objectName"]),
            on="object",
        )
        self.__ontology["subjectType"] = ["disease"] * len(self.__ontology)
        self.__ontology["objectType"] = ["disease"] * len(self.__ontology)
        self.__ontology["source"] = ["MONDO"] * len(self.__ontology)
        self.__ontology = (
            self.__ontology[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        self.__diseases = pd.concat(
            [
                self.__ontology[self.__ontology["subjectType"] == "disease"][
                    ["subject", "subjectName"]
                ].rename(
                    columns={"subject": "diseaseId", "subjectName": "diseaseName"}
                ),
                self.__ontology[self.__ontology["objectType"] == "disease"][
                    ["object", "objectName"]
                ].rename(columns={"object": "diseaseId", "objectName": "diseaseName"}),
            ]
        ).drop_duplicates(ignore_index=True)

        # dicts for info retrieval
        self.__mondo2name_asdict = dict(self.__mondo2name)

        log.info(f"{self.__class__.__name__} ready!")

    def get_MONDO(self, id):
        if id.startswith("OMIM"):
            return self.__omim2mondo.get(id)
        elif id.startswith("ORPHA"):
            return self.__orpha2mondo.get(id)
        elif id.startswith("DOID"):
            return self.__doid2mondo.get(id)

    @property
    def mondo2name(self):
        return self.__mondo2name.copy()

    def get_name(self, id):
        """Returns the name of a MONDO id"""
        if not id.startswith("MONDO:"):
            id = f"MONDO:{id}"
        return self.__mondo2name_asdict.get(id, f"Id {id} not found")

    @property
    def mondoRelationship(self):
        return self.__mondoRelationship.copy()

    @property
    def umls2mondo(self):
        return self.__umls2mondo.copy()

    @property
    def ontology(self):
        return self.__ontology.copy()

    @property
    def diseases(self):
        return self.__diseases.copy()


# @profile()
class DISEASES(Database, metaclass=Singleton):
    """
    DISEASES database by Jensen Lab reference class
    Contains info about diseases
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://diseases.jensenlab.org/Downloads",
            requirements=[NCBI, MONDO],
        )
        self.__db = self._add_file(
            url="https://download.jensenlab.org/human_disease_knowledge_filtered.tsv",
            names=[
                "geneId",
                "geneSymbol",
                "diseaseId",
                "diseaseName",
                "source",
                "evidence",
                "confidenceScore",
            ],
            usecols=["geneSymbol", "diseaseId"],
        )
        self.__db.content["diseaseId"] = self.__db.content["diseaseId"].map(
            self.__MONDO.get_MONDO
        )
        self.__db.content["geneSymbol"] = self.__db.content["geneSymbol"].map(
            self.__NCBI.check_symbol
        )
        self.__db.content.dropna(subset=["geneSymbol", "diseaseId"], inplace=True)
        self.__db.content.drop_duplicates(ignore_index=True, inplace=True)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()


# @profile()
class HPO(Database, metaclass=Singleton):
    """
    Human Phenotype Ontology reference class
    Contains info about phenotypes
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available (with conditions)",
            license_url="https://hpo.jax.org/app/license",
            requirements=[MONDO],
        )
        self.__db = self._add_file(url="http://purl.obolibrary.org/obo/hp.obo")
        self.__disease2phenotype = self._add_file(
            url="http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
            skiprows=4,
            rename_columns={"database_id": "diseaseId", "hpo_id": "phenotypeId"},
            query="evidence == 'TAS' and aspect == 'P'",
            final_columns=["diseaseId", "phenotypeId"],
        )
        self.__disease2phenotype.content[
            "diseaseId"
        ] = self.__disease2phenotype.content["diseaseId"].map(self.__MONDO.get_MONDO)
        self.__phenotype2name = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    self.__phenotype2name.add((f"HP:{frame.id.local}", clause.name))
        self.__disease2phenotype.content = self.__disease2phenotype.content.merge(
            pd.DataFrame(self.__MONDO.mondo2name, columns=["diseaseId", "diseaseName"]),
            on="diseaseId",
            how="left",
        )
        self.__disease2phenotype.content = (
            self.__disease2phenotype.content.merge(
                pd.DataFrame(
                    self.__phenotype2name, columns=["phenotypeId", "phenotypeName"]
                ),
                on="phenotypeId",
                how="left",
            )
            .dropna(subset=["diseaseName", "phenotypeName"])
            .drop_duplicates(ignore_index=True)
        )

        self.__gene2phenotype = self._add_file(
            url="http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
            # custom_read_function=read_gene2phenotype,
        )
        self.__geneSymbol2phenotype = {
            symbol: set(group["hpo_id"].values)
            for symbol, group in self.__gene2phenotype()[
                ["gene_symbol", "hpo_id"]
            ].groupby("gene_symbol")
        }
        self.__geneId2phenotype = {
            id: set(group["hpo_id"].values)
            for id, group in self.__gene2phenotype()[
                ["ncbi_gene_id", "hpo_id"]
            ].groupby("ncbi_gene_id")
        }

        # dicts for info retrieval
        self.__phenotype2name_asdict = dict(self.__phenotype2name)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def disease2phenotype(self):
        return self.__disease2phenotype().copy()

    @property
    def ontology(self):
        return self.__db().copy()

    @property
    def geneId2phenotype(self):
        return self.__geneId2phenotype.copy()

    @property
    def geneSymbol2phenotype(self):
        return self.__geneSymbol2phenotype.copy()

    def get_name(self, id):
        """Returns the name of a HPO id"""
        if not id.startswith("HP:"):
            id = f"HP:{id}"
        return self.__phenotype2name_asdict.get(id, f"Id {id} not found")


# @profile()
class GO(Database, metaclass=Singleton):
    """
    Gene Ontology reference class
    Contains info about biological processes,
    molecular functions,
    and cellular components
    """

    def __init__(self, basic=False, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="http://geneontology.org/docs/go-citation-policy/#license",
            requirements=[NCBI],
        )

        goa_columns = [
            "DB",
            "DB Object ID",
            "DB Object Symbol",
            "Qualifier",
            "GO ID",
            "DB Reference",
            "Evidence Code",
            "With (or) From",
            "Aspect",
            "DB Object Name",
            "DB Object Synonym",
            "DB Object Type",
            "Taxon",
            "Date",
            "Assigned By",
            "Annotatoin Extension",
            "Gene Product From ID",
        ]
        self.__gene_associations = self._add_file(
            url="http://geneontology.org/gene-associations/goa_human.gaf.gz",
            skiprows=41,
            names=goa_columns,
            usecols=[
                "DB Object Symbol",
                "Qualifier",
                "GO ID",
                "Evidence Code",
                "DB Object Synonym",
            ],
        )
        self.__gene_associations.content = self.__gene_associations.content[
            self.__gene_associations.content["Evidence Code"].isin(
                [
                    "EXP",
                    "IDA",
                    "IPI",
                    "IMP",
                    "IGI",
                    "IEP",
                    "HTP",
                    "HDA",
                    "HMP",
                    "HGI",
                    "HEP",
                    "IBA",
                    "IBD",
                    "IKR",
                    "IRD",
                    "TAS",
                    "IC",
                    "NAS",
                ]
            )
        ].reset_index(drop=True)
        self.__gene2go = set()
        for symbol, qualifier, goid, synonyms in tqdm(
            self.__gene_associations()[
                ["DB Object Symbol", "Qualifier", "GO ID", "DB Object Synonym"]
            ].itertuples(index=False),
            desc=f"Collecting data from {self.__gene_associations.filename}",
        ):
            if isinstance(synonyms, str):  # in order to avoid nans
                symbol = self.__NCBI.check_symbol(symbol, synonyms.split("|"))
            else:
                symbol = self.__NCBI.check_symbol(symbol)
            if symbol:
                self.__gene2go.add((symbol, qualifier, goid))
        self.__gene2go = pd.DataFrame(
            self.__gene2go, columns=["geneSymbol", "relation", "goTerm"]
        )

        def fix_go_obofile(filepath):
            """
            Tries to fix the go go.obo file to be readable by fastobo
            """
            content = []
            with open(filepath, "r") as infile:
                for line in infile:
                    if line.split(":")[0] in ["synonym", "def"]:
                        try:
                            s = re.findall(
                                '[a-zA-Z]+:\ ".+"\ [A-Z \s _]*\[(.+)\]|$', line.strip()
                            )[0]
                            if len(s):
                                line = line.replace(s, s.replace(" ", "_"))
                        except:
                            pass
                    elif (
                        line.split(":")[0] == "xref" and line.count(":") == 1
                    ):  # drop xref lines with non-standard format
                        line = ""
                    elif (
                        line.split(":")[0] == "xref" and line.count(":") > 1
                    ):  # removes space between database code and relative id in xref
                        line = f'xref: {line.split("xref: ")[-1].replace(": ", ":")}'
                    content.append(line)
            filepath = os.path.join(
                os.path.dirname(filepath), f"cleaned-{os.path.basename(filepath)}"
            )
            with open(filepath, "w") as outfile:
                for line in content:
                    outfile.write(line)
            return filepath

        if basic:
            self.__db = self._add_file(
                url="http://geneontology.org/ontology/go-basic.obo",
                fix_function=fix_go_obofile,
            )
        else:
            self.__db = self._add_file(
                url="http://geneontology.org/ontology/go.obo",
                fix_function=fix_go_obofile,
            )
        self.__go2name = set()
        self.__goRelationship = set()
        self.__go2namespace = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            is_obsolete = False
            relationship_tmp = []
            for clause in frame:
                if isinstance(clause, fastobo.term.IsObsoleteClause):
                    is_obsolete = True
                    break
                elif isinstance(clause, fastobo.term.NameClause):
                    self.__go2name.add((str(frame.id), clause.name))
                elif isinstance(clause, fastobo.term.RelationshipClause):
                    relationship_tmp.append(
                        (str(frame.id), clause.typedef.escaped, str(clause.term))
                    )
                elif isinstance(clause, fastobo.term.IsAClause):
                    relationship_tmp.append((str(frame.id), "is_a", str(clause.term)))
                elif isinstance(clause, fastobo.term.NamespaceClause):
                    self.__go2namespace.add(
                        (str(frame.id), camelize(str(clause.namespace)))
                    )
            if not is_obsolete:
                self.__goRelationship.update(relationship_tmp)

        self.__gene2go = self.__gene2go.merge(
            pd.DataFrame(self.__go2name, columns=["goTerm", "goName"]),
            on="goTerm",
            how="left",
        )
        self.__gene2go = (
            self.__gene2go.merge(self.__NCBI.gene2name, on="geneSymbol")
            .dropna(subset=["geneName"])
            .reset_index(drop=True)
        )
        self.__gene2go = (
            self.__gene2go.merge(
                pd.DataFrame(self.__go2namespace, columns=["goTerm", "goNamespace"]),
                on="goTerm",
                how="left",
            )
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "geneSymbol": "string",
                    "relation": "category",
                    "goTerm": "string",
                    "goName": "string",
                    "geneName": "string",
                    "goNamespace": "category",
                }
            )
        )

        # keep local (only needed for building the ontology)
        gene2go = self.__gene2go.rename(
            columns={
                "geneSymbol": "subject",
                "goTerm": "object",
                "geneName": "subjectName",
                "goName": "objectName",
            }
        )
        gene2go["subjectType"] = ["protein"] * len(gene2go)
        gene2go.insert(len(gene2go.columns), "source", self.__class__.__name__)
        gene2go = (
            gene2go.merge(
                pd.DataFrame(self.__go2namespace, columns=["object", "objectType"]),
                on="object",
                how="left",
            )
            .dropna(subset=["objectType"])
            .reset_index(drop=True)
        )
        gene2go = gene2go[
            [
                "subject",
                "relation",
                "object",
                "subjectName",
                "objectName",
                "subjectType",
                "objectType",
                "source",
            ]
        ].astype(
            {
                "subject": "string",
                "relation": "category",
                "object": "string",
                "subjectName": "string",
                "objectName": "string",
                "subjectType": "category",
                "objectType": "category",
                "source": "string",
            }
        )

        self.__ontology = pd.DataFrame(
            self.__goRelationship, columns=["subject", "relation", "object"]
        )
        for column in ["subject", "object"]:
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__go2name, columns=[column, f"{column}Name"]),
                on=column,
                how="left",
            )
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__go2namespace, columns=[column, f"{column}Type"]),
                on=column,
                how="left",
            )
        self.__ontology.insert(
            len(self.__ontology.columns), "source", self.__class__.__name__
        )
        self.__ontology = (
            self.__ontology[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )
        self.__ontology = pd.concat([self.__ontology, gene2go]).drop_duplicates(
            ignore_index=True
        )

        # for every namespace set a property to retrieve ids and names
        namespaces = {t[1] for t in self.__go2namespace}
        for namespace in namespaces:
            value = pd.concat(
                [
                    self.__ontology[self.__ontology["subjectType"] == namespace][
                        ["subject", "subjectName"]
                    ].rename(
                        columns={
                            "subject": namespace,
                            "subjectName": f"{namespace}Name",
                        }
                    ),
                    self.__ontology[self.__ontology["objectType"] == namespace][
                        ["object", "objectName"]
                    ].rename(
                        columns={"object": namespace, "objectName": f"{namespace}Name"}
                    ),
                ]
            )
            setattr(
                self.__class__,
                inflection.pluralize(namespace),
                property(lambda x: value.copy()),
            )  # uselessy too complex? maybe

        # dicts for info retrieval
        self.__go2name_asdict = dict(self.__go2name)
        self.__go2namespace_asdict = dict(self.__go2namespace)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def gene2go(self):
        return self.__gene2go.copy()

    @property
    def go2name(self):
        return self.__go2name.copy()

    @property
    def go2namespace(self):
        return self.__go2namespace.copy()

    @property
    def ontology(self):
        return self.__ontology.copy()

    def get_name(self, id):
        """Returns the name of a GO id"""
        if not id.startswith("GO:"):
            id = f"GO:{id}"
        return self.__go2name_asdict.get(id, f"Id {id} not found")

    def get_namespace(self, id):
        """Returns the namespace associated to a GO id"""
        if not id.startswith("GO:"):
            id = f"GO:{id}"
        return self.__go2namespace_asdict.get(id, f"Id {id} not found")


# @profile()
class PathwayCommons(Database, metaclass=Singleton):
    """
    PathwayCommons database reference class
    Contains info about pathways
    """

    def __init__(self, update=None):
        from bs4 import BeautifulSoup

        Database.__init__(
            self,
            update=update,
            license="Freely available, under the license terms of each contributing database (https://www.pathwaycommons.org/pc2/datasources)",
            license_url="https://www.pathwaycommons.org/pc/about.do",
            requirements=[NCBI, UniProt],
        )
        url = "https://www.pathwaycommons.org/archives/PC2/"
        latest = max(
            [
                int(re.findall("v([0-9]+)/", h.get_text())[0])
                for h in BeautifulSoup(requests.get(url).content, "html5lib").findAll(
                    name="a"
                )
                if "v" in h.get_text()
            ]
        )

        def read_pathwaycommons_gmt(filepath):
            import gzip
            import pandas as pd
            import re

            datasources = {  # for nicer (and uniform) formattation
                "panther": "PANTHER",
                "kegg": "KEGG",
                "inoh": "INOH",
                "netpath": "NetPath",
                "pathbank": "PathBank",
                "pid": "PID",
                "reactome": "Reactome",
                "humancyc": "HumanCyc",
            }
            gene2pathway = set()
            with gzip.open(filepath, "rt") as infile:
                for line in infile:
                    """
                    Every line contains an unique identifier (url) for the geneset,
                    a description of the geneset,
                    and the genes in the geneset
                    """
                    line = line.strip().split("\t")
                    id = line[0].split("/")[-1]
                    name, datasource, organism, idtype = re.findall(
                        "name: (.*); datasource: (.*); organism: (.*); idtype: (.*)",
                        line[1],
                    )[0]
                    genes = line[2:]
                    if (
                        organism == "9606" and name != "untitled"
                    ):  # for checking that all are about homo sapiens and have a name
                        for gene in genes:
                            gene2pathway.add(
                                (
                                    gene,
                                    f"{datasources[datasource]}:{id}",
                                    name,
                                    f"PathwayCommons:{datasources[datasource]}",
                                )
                            )
                gene2pathway = pd.DataFrame(
                    gene2pathway,
                    columns=["uniprotId", "pathwayId", "pathwayName", "source"],
                )

                return gene2pathway

        self.__gene2pathway = self._add_file(
            url=f"{url}v{latest}/PathwayCommons{latest}.All.uniprot.gmt.gz",
            retrieved_version=f"v{latest}",
            custom_read_function=read_pathwaycommons_gmt,
        )
        self.__gene2pathway.content = (
            self.__gene2pathway.content.merge(
                self.__UniProt.uniprot2symbol[["uniprotId", "geneSymbol"]],
                how="left",
                on="uniprotId",
            )
            .dropna()
            .reset_index(drop=True)[
                ["geneSymbol", "pathwayId", "pathwayName", "source"]
            ]
        )

        # keep local (only needed for building the ontology)
        self.__db = self.__gene2pathway().rename(
            columns={
                "geneSymbol": "subject",
                "pathwayId": "object",
                "pathwayName": "objectName",
            }
        )
        self.__db = (
            self.__db.merge(
                self.__NCBI.gene2name.rename(
                    columns={"geneSymbol": "subject", "geneName": "subjectName"}
                ),
                on="subject",
            )
            .dropna(subset=["subjectName"])
            .reset_index(drop=True)
        )
        self.__db["subjectType"] = ["protein"] * len(self.__db)
        self.__db["objectType"] = ["pathway"] * len(self.__db)
        self.__db["relation"] = ["participates_in"] * len(
            self.__db
        )  # to check # takes_part_in involved_in
        self.__db = (
            self.__db[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        self.__pathways = (
            self.__gene2pathway()[["pathwayId", "pathwayName"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db.copy()

    @property
    def gene2pathway(self):
        return self.__gene2pathway().copy()

    @property
    def pathways(self):
        return self.__pathways.copy()


# @profile()
class Bgee(Database, metaclass=Singleton):
    """
    Bgee reference class
    Contains info about gene expression patterns
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC0 1.0",
            license_url="https://bgee.org/?page=about",
            requirements=[NCBI, Uberon],
        )
        self.__db = self._add_file(
            url="ftp://ftp.bgee.org/current/download/calls/expr_calls/Homo_sapiens_expr_simple.tsv.gz",
            usecols=["Expression", "Gene name", "Anatomical entity ID", "Call quality"],
            query="`Expression` == 'present' and `Call quality` == 'gold quality'",  # only best quality https://www.bgee.org/support/gene-expression-calls#final-step-generation-of-presentabsent-expression-calls-per-gene-and-condition
            rename_columns={
                "Gene name": "geneSymbol",
                "Anatomical entity ID": "anatomicalEntityId",
            },
            final_columns=["geneSymbol", "anatomicalEntityId"],
        )
        self.__db.content["geneSymbol"] = self.__db.content["geneSymbol"].map(
            self.__NCBI.check_symbol
        )
        self.__db.content = self.__db.content.dropna(subset=["geneSymbol"]).reset_index(
            drop=True
        )

        self.__gene_expression = self.__db()[
            ~self.__db()["anatomicalEntityId"].str.startswith("PBA:")
        ]  # discards tissues relative to primates (NCBITaxon:9443)

        self.__gene_expression = (
            self.__gene_expression.merge(
                pd.DataFrame(
                    self.__Uberon.uberon2name,
                    columns=["anatomicalEntityId", "anatomicalEntityName"],
                ),
                on="anatomicalEntityId",
                how="left",
            )
            .dropna()
            .reset_index(drop=True)
        )

        self.__gene_expression = (
            self.__gene_expression.merge(
                self.__NCBI.gene2name, on="geneSymbol", how="left",
            )
            .dropna(subset=["geneSymbol"])
            .reset_index(drop=True)
        )

        self.__gene_expression = (
            self.__gene_expression.merge(
                pd.DataFrame(
                    self.__Uberon.uberon2namespace,
                    columns=["anatomicalEntityId", "anatomicalEntityType"],
                ),
                on="anatomicalEntityId",
                how="left",
            )
            .dropna()
            .reset_index(drop=True)
        )

        self.__gene_expression = (
            self.__gene_expression[
                [
                    "geneSymbol",
                    "anatomicalEntityId",
                    "geneName",
                    "anatomicalEntityName",
                    "anatomicalEntityType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "geneSymbol": "string",
                    "anatomicalEntityId": "string",
                    "geneName": "string",
                    "anatomicalEntityName": "string",
                    "anatomicalEntityType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def gene_expression(self):
        return self.__gene_expression.copy()


# @profile()
class Uberon(Database, metaclass=Singleton):
    """
    Uberon reference class
    Contains info about anatomical entities
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC-BY 3.0",
            license_url="https://github.com/obophenotype/uberon/issues/1139",
            requirements=[PRO, GO],
        )

        def fix_uberon_obofile(filepath):
            """
            Tries to fix the uberon human-view.obo file to be readable by fastobo
            """
            content = []
            with open(filepath, "r") as infile:
                for line in infile:
                    if line.split(":")[0] in ["synonym", "def"]:
                        try:
                            s = re.findall(
                                '[a-zA-Z]+:\ ".+"\ [A-Z \s _]*\[(.+)\]|$', line.strip()
                            )[0]
                            if len(s):
                                line = line.replace(s, s.replace(" ", "_"))
                        except:
                            pass
                    elif (
                        line.split(":")[0] == "xref" and line.count(":") == 1
                    ):  # drop xref lines with non-standard format
                        line = ""
                    content.append(line)
            filepath = os.path.join(
                os.path.dirname(filepath), f"cleaned-{os.path.basename(filepath)}"
            )
            with open(filepath, "w") as outfile:
                for line in content:
                    outfile.write(line)
            return filepath

        self.__db = self._add_file(
            url="http://purl.obolibrary.org/obo/uberon/subsets/human-view.obo",
            fix_function=fix_uberon_obofile,
        )

        relationships2discard = (
            "ambiguous_for_taxon",
            "dc-contributor",
            "dc-creator",
            "depicted_by",
            "dubious_for_taxon",
            "fma_set_term",
            "implements_design_pattern",
            "in_taxon",
            "only_in_taxon",
            "present_in_taxon",
            "seeAlso",
            "source_atlas",
        )
        self.__uberonIsA = set()
        self.__uberonRelationship = set()
        self.__uberon2name = set()
        self.__uberon2namespace = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            if hasattr(frame.id, "prefix") and frame.id.prefix in [
                "CL",
                "GO",
                "NBO",
                "PR",
                "UBERON",
            ]:
                is_obsolete = False
                name = None
                relationships = []
                isA = []
                namespace = []
                for clause in frame:
                    if isinstance(clause, fastobo.term.IsObsoleteClause):
                        is_obsolete = True
                        break
                    elif isinstance(clause, fastobo.term.NameClause):
                        name = clause.name
                    elif isinstance(
                        clause, fastobo.term.IsAClause
                    ) and clause.term.prefix in ["CL", "GO", "NBO", "PR", "UBERON",]:
                        isA.append((str(frame.id), "is_a", str(clause.term)))
                    elif (
                        isinstance(clause, fastobo.term.RelationshipClause)
                        and str(clause.typedef) not in relationships2discard
                        and clause.term.prefix in ["CL", "GO", "NBO", "PR", "UBERON"]
                    ):
                        relationships.append(
                            (str(frame.id), str(clause.typedef), str(clause.term))
                        )
                    elif isinstance(clause, fastobo.term.NamespaceClause):
                        namespace = camelize(str(clause.namespace))
                        if namespace == "cl":
                            namespace = "cell"
                if not is_obsolete:
                    self.__uberonIsA.update(isA)
                    self.__uberonRelationship.update(relationships)
                    if name:
                        self.__uberon2name.add((str(frame.id), name))
                    if namespace:
                        self.__uberon2namespace.add((str(frame.id), namespace))

        self.__ontology = pd.DataFrame(
            self.__uberonIsA, columns=["subject", "relation", "object"]
        )
        self.__ontology = pd.concat([
            self.__ontology,
            pd.DataFrame(
                self.__uberonRelationship, columns=["subject", "relation", "object"]
            )],
            ignore_index=True,
        )
        for role in ["subject", "object"]:
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__PRO.pro2name, columns=[role, f"{role}Name"]),
                on=role,
                how="left",
            ).reset_index(
                drop=True
            )  # gets names for PRO entities (their Uberon names not always are concordant with PRO)

            self.__ontology[f"{role}Name"] = self.__ontology[f"{role}Name"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame(self.__uberon2name, columns=[role, f"{role}Name"]),
                    on=role,
                    how="left",
                )[f"{role}Name"]
            )  # then fills NaNs with those retrieved from Uberon file

            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__uberon2namespace, columns=[role, f"{role}Type"]),
                on=role,
                how="left",
            ).reset_index(drop=True)

            self.__ontology[f"{role}Type"] = self.__ontology[f"{role}Type"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame(
                        self.__PRO.pro2category, columns=[role, f"{role}Type"]
                    ).replace("gene", "protein"),
                    on=role,
                    how="left",
                )[f"{role}Type"]
            )  # fills Type for PR objects

            self.__ontology[f"{role}Type"] = self.__ontology[f"{role}Type"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame(
                        self.__GO.go2namespace, columns=[role, f"{role}Type"]
                    ).replace("gene", "protein"),
                    on=role,
                    how="left",
                )[f"{role}Type"]
            )  # fills Type for GO objects

            for db, namespace in {
                ("UBERON", "anatomicalEntity"),
                ("CARO", "anatomicalEntity"),
                ("NBO", "behaviour"),
                ("CL", "cell"),
            }:  # fills Type for listed databases
                self.__ontology[f"{role}Type"] = self.__ontology[f"{role}Type"].fillna(
                    self.__ontology[[role]].merge(
                        self.__ontology[self.__ontology[role].str.startswith(f"{db}:")]
                        .fillna(namespace)[[role, f"{role}Type"]]
                        .drop_duplicates(ignore_index=True),
                        on=role,
                        how="left",
                    )[f"{role}Type"]
                )
            self.__ontology = (
                self.__ontology[
                    ~self.__ontology[f"{role}Type"].str.contains(
                        "[M,m]odification|[S,s]eqgroup|[S,s]equence", na=False
                    )
                ]
                .reset_index(drop=True)
                .replace(
                    {
                        "family": "proteinFamily",
                        "organismFamily": "organismProteinFamily",
                    }
                )
            )  # for the sake of clarity

        self.__ontology["source"] = ["Uberon"] * len(self.__ontology)
        self.__ontology = (
            self.__ontology.dropna()
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        self.__uberon2namespace = set(
            zip(self.__ontology["subject"], self.__ontology["subjectType"])
        )
        self.__uberon2namespace.update(
            set(zip(self.__ontology["object"], self.__ontology["objectType"]))
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def ontology(self):
        return self.__ontology.copy()

    @property
    def uberonRelationship(self):
        return self.__uberonRelationship.copy()

    @property
    def uberon2name(self):
        return self.__uberon2name.copy()

    @property
    def uberon2namespace(self):
        return self.__uberon2namespace.copy()


# @profile()
class APID(Database, metaclass=Singleton):
    """
        APID reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC-BY-NC",
            license_url="http://cicblade.dep.usal.es:8080/APID/init.action#subtab4",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://cicblade.dep.usal.es:8080/APID/InteractionsTABplain.action",
            post=True,
            data_to_send={
                "interactomeTaxon": "9606",
                "interactomeTaxon1": "9606",
                "quality": 1,
                "interspecies": "NO",
            },
            final_columns=["GeneName_A", "GeneName_B"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions["subject"] = self.__interactions["GeneName_A"].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["GeneName_B"].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class BioGRID(Database, metaclass=Singleton):
    """
        BioGRID reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="MIT",
            license_url="https://downloads.thebiogrid.org/BioGRID",
            requirements=[NCBI],
        )

        import requests
        from bs4 import BeautifulSoup

        physical_experimental_systems = [
            tr.th.text.lstrip(" ").rstrip(" ")
            for tr in BeautifulSoup(
                requests.get(
                    "https://wiki.thebiogrid.org/doku.php/curation_guide:biochemical_experimental_systems"
                ).content,
                "html5lib",
            )
            .find("table")
            .findAll("tr")[1:]
        ]

        def read_biogrid_file(filepath):
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open(f"BIOGRID-ORGANISM-Homo_sapiens-{self.v}.tab.txt") as f:
                    return pd.read_csv(f, skiprows=35, sep="\t", dtype=str).query(
                            "ORGANISM_A_ID == '9606' and ORGANISM_B_ID == '9606'"
                        ).query(
                            f"EXPERIMENTAL_SYSTEM in {physical_experimental_systems}"
                        )

        self.__db = self._add_file(
            url=self.get_current_release_url(),
            retrieved_version=self.v,  # retrieved by get_current_release_url
            custom_read_function=read_biogrid_file,
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions = self.__interactions.query(
            # only low-throughput studies (Highest PubMed Identifier (PMID) Reuse < 20)
            f"PUBMED_ID in {self.__interactions['PUBMED_ID'].value_counts().where(self.__interactions['PUBMED_ID'].value_counts()<20).dropna().index.tolist()}"
        )[
            [
                "OFFICIAL_SYMBOL_A",
                "OFFICIAL_SYMBOL_B",
                "ALIASES_FOR_A",
                "ALIASES_FOR_B",
                "source",
            ]
        ]
        self.__interactions["subject"] = self.__interactions[
            ["OFFICIAL_SYMBOL_A", "ALIASES_FOR_A"]
        ].apply(
            lambda x: self.__NCBI.check_symbol(x[0], x[1].split("|")), axis=1
        )  # checks symbol A symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            ["OFFICIAL_SYMBOL_B", "ALIASES_FOR_B"]
        ].apply(
            lambda x: self.__NCBI.check_symbol(x[0], x[1].split("|")), axis=1
        )  # hecks symbol B symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = list(sources_data["BioGRID"]["files"].keys())[0]
                    url = sources_data["BioGRID"]["files"][filename]["URL"]
                    version = sources_data["BioGRID"]["files"][filename]["version"]
                    self.v = re.findall("(.+)\    ", version)[
                        0
                    ]  # before four spaces (tab)
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            from bs4 import BeautifulSoup

            r = requests.get("https://downloads.thebiogrid.org/BioGRID/")
            page = r.content
            soup = BeautifulSoup(page, "html5lib")
            href = soup.find("a", string="Current-Release")["href"]
            self.v = re.findall("BIOGRID-(.+)/", href)[0]
            url = f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{self.v}/BIOGRID-ORGANISM-{self.v}.tab.zip"
            return url


# @profile()
class HPRD(Database, metaclass=Singleton):
    """
        HPRD reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available for non-commercial purposes",
            license_url="https://hprd.org/FAQ/index_html",
            requirements=[NCBI],
        )

        def read_hprd_file(filepath):
            import tarfile

            with tarfile.open(filepath) as archive:
                return pd.read_csv(
                    archive.extractfile(
                        [
                            file
                            for file in archive.getnames()
                            if file.endswith("BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt")
                        ][0]
                    ),
                    sep="\t",
                    names=[
                        "interactor_1_geneSymbol",
                        "interactor_1_hprd_id",
                        "interactor_1_refseq_id",
                        "interactor_2_geneSymbol",
                        "interactor_2_hprd_id",
                        "interactor_2_refseq_id",
                        "experiment_type",
                        "reference_id",
                    ],
                    dtype="category",
                )

        self.__db = self._add_file(
            url="http://hprd.org/RELEASE9/HPRD_FLAT_FILES_041310.tar.gz",
            custom_read_function=read_hprd_file,
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        # only low-throughput studies (Highest PubMed Identifier (PMID) Reuse < 20)
        self.__interactions = self.__interactions[
            [
                "interactor_1_geneSymbol",
                "interactor_1_hprd_id",
                "interactor_1_refseq_id",
                "interactor_2_geneSymbol",
                "interactor_2_hprd_id",
                "interactor_2_refseq_id",
                "experiment_type",
                "source",
            ]
        ].join(self.__interactions.reference_id.str.split(",").explode())
        self.__interactions = self.__interactions.query(
            f"reference_id in {self.__interactions['reference_id'].value_counts().where(self.__interactions['reference_id'].value_counts()<20).dropna().index.tolist()}"
        )[
            ["interactor_1_geneSymbol", "interactor_2_geneSymbol", "source"]
        ].drop_duplicates(
            ignore_index=True
        )

        self.__interactions["subject"] = self.__interactions[
            "interactor_1_geneSymbol"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            "interactor_2_geneSymbol"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = [
            "interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = [
            "protein"] * len(self.__interactions)
        self.__interactions["objectType"] = [
            "protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class HuRI(Database, metaclass=Singleton):
    """
        HuRI reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="http://www.interactome-atlas.org/download",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://www.interactome-atlas.org/data/HuRI.tsv",
            names=["protein1", "protein2"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions = self.__interactions.merge(
            self.__NCBI.ensembl2ncbi.rename(
                columns={"Ensembl": "protein1", "NCBI": "subject"}
            ),
            on="protein1",
            how="left",
        )  # converts Ensembl ids to NCBI
        self.__interactions["subject"] = self.__interactions["subject"].apply(
            self.__NCBI.check_symbol
        )  # checks protein1 symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions = self.__interactions.merge(
            self.__NCBI.ensembl2ncbi.rename(
                columns={"Ensembl": "protein2", "NCBI": "object"}
            ),
            on="protein2",
            how="left",
        )  # converts Ensembl ids to NCBI
        self.__interactions["object"] = self.__interactions["object"].apply(
            self.__NCBI.check_symbol
        )  # checks protein2 symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class InnateDB(Database, metaclass=Singleton):
    """
        InnateDB reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="DESIGN SCIENCE LICENSE",
            license_url="https://www.innatedb.com/license.jsp",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="https://www.innatedb.com/download/interactions/all.mitab.gz",
            usecols=[
                "alias_A",
                "alias_B",
                "ncbi_taxid_A",
                "ncbi_taxid_B",
                "confidence_score",
            ],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db().query(
            "ncbi_taxid_A == 'taxid:9606(Human)' and ncbi_taxid_B == 'taxid:9606(Human)'"
        )  # check if actually from Homo sapiens
        self.__interactions = self.__interactions[
            self.__interactions["confidence_score"].apply(
                lambda s: int(re.findall("np:([0-9]+)\|", s)[0]) >= 1
            )
        ]  # checks that there is at least one publication supporting the interaction that has never been used to support any other interaction #http://wodaklab.org/iRefWeb/faq
        self.__interactions["subject"] = self.__interactions["alias_A"].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall("hgnc:(.+)\(display_short\)|$", s)[0]
            )
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["alias_B"].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall("hgnc:(.+)\(display_short\)|$", s)[0]
            )
        )  # checks the symbol of the second gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = self.__interactions[["subject", "object", "source"]]
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class INstruct(Database, metaclass=Singleton):
    """
        INstruct reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="All rights reserved (Authorization obtained by e-mail contact with Haiyuan Yu <haiyuan.yu@cornell.edu>)",
            license_url="http://instruct.yulab.org/about.html",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://instruct.yulab.org/download/sapiens.sin",
            usecols=["ProtA[Official Symbol]", "ProtB[Official Symbol]"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions["subject"] = self.__interactions[
            "ProtA[Official Symbol]"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks protA symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            "ProtB[Official Symbol]"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks protB symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class IntAct(Database, metaclass=Singleton):
    """
        IntAct reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC-BY 4.0",
            license_url="https://www.ebi.ac.uk/intact/resources/overview",
            requirements=[NCBI],
        )

        def read_intact_file(filepath):
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open("intact.txt") as f:
                    return pd.read_csv(f, sep="\t", dtype=str)

        self.__db = self._add_file(
            url="http://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip",
            custom_read_function=read_intact_file,
            usecols=[
                "Alias(es) interactor A",
                "Alias(es) interactor B",
                "Taxid interactor A",
                "Taxid interactor B",
                "Confidence value(s)",
            ],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions = self.__interactions[
            (
                self.__interactions["Taxid interactor A"].str.contains(
                    "taxid:9606\(Homo sapiens\)"
                )
            )
            & (
                self.__interactions["Taxid interactor B"].str.contains(
                    "taxid:9606\(Homo sapiens\)"
                )
            )
        ]  # check if actually from Homo sapiens (both interactor A and B)
        self.__interactions = self.__interactions[
            self.__interactions["Confidence value(s)"].apply(
                lambda s: float(re.findall("intact-miscore:(.+)", s)[0]) >= 0.6
            )
        ]  # threshold for high confidence https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4316181/pdf/bau131.pdf
        self.__interactions["subject"] = self.__interactions[
            "Alias(es) interactor A"
        ].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall(":([a-zA-Z]+)\(gene name\)|$", s)[0],
                [
                    re.findall(":(.+)\(|$", alias)[0]
                    for alias in s.split("|")
                    if "(gene name synonym)" in alias
                ],
            )
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            "Alias(es) interactor B"
        ].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall(":([a-zA-Z]+)\(gene name\)|$", s)[0],
                [
                    re.findall(":(.+)\(|$", alias)[0]
                    for alias in s.split("|")
                    if "(gene name synonym)" in alias
                ],
            )
        )  # checks the symbol of the second gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = self.__interactions[
            ["subject", "object", "source"]
        ].drop_duplicates(ignore_index=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class PINA(Database, metaclass=Singleton):
    """
        PINA2 reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Downloable - All Rights Reserved (check with the develop team https://omics.bjcancer.org/pina2012/contact.do)",
            license_url="https://doi.org/10.1093%2Fnar%2Fgkr967 https://omics.bjcancer.org/pina2012/interactome.stat.do",
            requirements=[NCBI, UniProt],
        )
        self.__db = self._add_file(
            url="https://omics.bjcancer.org/pina2012/download/Homo%20sapiens-20140521.tsv",
            usecols=[
                "ID(s) interactor A",
                "ID(s) interactor B",
                "Interaction detection method(s)",
                "Publication Identifier(s)",
                "Taxid interactor A",
                "Taxid interactor B",
                "Interaction type(s)",
                "Source database(s)",
            ],
            dtype="category",
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        # check if actually from Homo sapiens
        self.__interactions = self.__interactions[
            self.__interactions["Taxid interactor A"] == "taxid:9606(Homo sapiens)"
        ]
        self.__interactions = self.__interactions[
            self.__interactions["Taxid interactor B"] == "taxid:9606(Homo sapiens)"
        ]
        self.__interactions = self.__interactions[
            [
                "ID(s) interactor A",
                "ID(s) interactor B",
                "Interaction detection method(s)",
                "Taxid interactor A",
                "Taxid interactor B",
                "Interaction type(s)",
                "Source database(s)",
                "source",
            ]
        ].join(
            self.__interactions["Publication Identifier(s)"].str.split(
                "|").explode()
        )

        self.__interactions = self.__interactions[
            self.__interactions["Publication Identifier(s)"].isin(
                self.__interactions["Publication Identifier(s)"]
                .value_counts()
                .where(
                    self.__interactions["Publication Identifier(s)"].value_counts(
                    ) < 20
                )
                .dropna()
                .index.tolist()
            )
        ][["ID(s) interactor A", "ID(s) interactor B", "source"]].drop_duplicates(
            ignore_index=True
        )
        self.__interactions["ID(s) interactor A"] = self.__interactions[
            "ID(s) interactor A"
        ].str.lstrip("uniprotkb:")
        self.__interactions["ID(s) interactor B"] = self.__interactions[
            "ID(s) interactor B"
        ].str.lstrip("uniprotkb:")
        self.__interactions = (
            self.__interactions.merge(
                self.__UniProt.uniprot2symbol[["uniprotId", "geneSymbol"]],
                how="left",
                left_on="ID(s) interactor A",
                right_on="uniprotId",
            )
            .dropna()
            .reset_index(drop=True)
            .rename(columns={"geneSymbol": "subject"})[
                ["subject", "ID(s) interactor B", "source"]
            ]
        )
        self.__interactions = (
            self.__interactions.merge(
                self.__UniProt.uniprot2symbol[["uniprotId", "geneSymbol"]],
                how="left",
                left_on="ID(s) interactor B",
                right_on="uniprotId",
            )
            .dropna()
            .reset_index(drop=True)
            .rename(columns={"geneSymbol": "object"})[["subject", "object", "source"]]
        )
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = [
            "interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = [
            "protein"] * len(self.__interactions)
        self.__interactions["objectType"] = [
            "protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class SignaLink(Database, metaclass=Singleton):
    """
        SignaLink reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC-SA 3.0",
            license_url="http://signalink.org/faq",
            requirements=[NCBI],
        )

        def read_signalink_file(filepath):
            import tarfile
            import json

            with tarfile.open(filepath) as archive:
                edges_member = [file for file in archive.getnames() if "edges" in file][
                    0
                ]
                nodes_member = [file for file in archive.getnames() if "nodes" in file][
                    0
                ]
                raw_edges = json.load(archive.extractfile(edges_member))
                raw_nodes = json.load(archive.extractfile(nodes_member))
            taxon = {
                node["displayedName"]: int(node["taxon"]["id"]) for node in raw_nodes
            }
            accepted_dbs = (
                "SignaLink",
                "ACSN",
                # "InnateDB",
                "Signor",
                # "PhosphoSite",
                # "TheBiogrid",
                # "ComPPI",
                "HPRD",
                # "IntAct",
                # "OmniPath",
            )  # PSP == Potential Scaffold Proteins
            edges = set()
            for edge in raw_edges:
                if len(
                    {
                        db["value"]
                        for db in edge["sourceDatabases"]
                        if db["value"] in accepted_dbs
                    }
                ):  # only trusted databases for proteins
                    source = edge["sourceDisplayedName"]
                    target = edge["targetDisplayedName"]
                    if (
                        taxon[source] == 9606 and taxon[target] == 9606
                    ):  # filters for homo sapiens
                        source = self.__NCBI.check_symbol(
                            source
                        )  # checks the symbol of the source
                        target = self.__NCBI.check_symbol(
                            target
                        )  # checks the symbol of the target
                        if source and target:
                            edges.add(
                                (
                                    source,
                                    target,
                                    edge["sourceFullName"],
                                    edge["targetFullName"],
                                    ", ".join(
                                        {db["value"] for db in edge["sourceDatabases"]}
                                    ),
                                    "|".join(
                                        [pmid["value"]
                                            for pmid in edge["publications"]]
                                    ),
                                )
                            )
            return (
                pd.DataFrame(
                    edges,
                    columns=[
                        "sourceSymbol",
                        "targetSymbol",
                        "sourceName",
                        "targetName",
                        "database",
                        "pmid",
                    ],
                )
                .drop_duplicates(ignore_index=True)
                .astype(
                    {
                        "sourceSymbol": "string",
                        "targetSymbol": "string",
                        "sourceName": "string",
                        "targetName": "string",
                        "database": "category",
                        "pmid": "string",
                    }
                )
            )

        self.__db = self._add_file(
            url="http://signalink.org/slk3db_dump_json.tgz",
            custom_read_function=read_signalink_file,
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        # only low-throughput studies (Highest PubMed Identifier (PMID) Reuse < 20)
        self.__interactions = self.__db()[
            [
                "sourceSymbol",
                "targetSymbol",
                "sourceName",
                "targetName",
                "database",
                "source",
            ]
        ].join(self.__db().pmid.str.split("|").explode())
        self.__interactions = self.__interactions.query(
            f"pmid in {self.__interactions['pmid'].value_counts().where(self.__interactions['pmid'].value_counts()<20).dropna().index.tolist()}"
        )[["sourceSymbol", "targetSymbol", "source"]].drop_duplicates(ignore_index=True)
        self.__interactions = self.__interactions.rename(
            columns={"sourceSymbol": "subject", "targetSymbol": "object"}
        )[["subject", "object", "source"]]
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name (names not provided by NCBI or HGNC are not trusted)
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name (names not provided by NCBI or HGNC are not trusted)
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class STRING(Database, metaclass=Singleton):
    """
        STRING reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://string-db.org/cgi/access?footer_active_subpage=licensing",
            requirements=[NCBI],
        )
        self.__symbol2string = self._add_file(
            url="https://string-db.org/mapping_files/STRING_display_names/human.name_2_string.tsv.gz",
            skiprows=1,
            names=["NCBI taxid", "geneSymbol", "STRING"],
            final_columns=["geneSymbol", "STRING"],
        )
        self.__symbol2string_asdict = (
            self.__symbol2string().set_index("geneSymbol")["STRING"].to_dict()
        )
        self.__string2symbol_asdict = (
            self.__symbol2string().set_index("STRING")["geneSymbol"].to_dict()
        )
        self.__db = self._add_file(
            url=self.get_current_release_url(),
            sep=" ",
            retrieved_version=self.v,  # retrieved by get_current_release_url
            usecols=["protein1", "protein2", "experiments", "combined_score"],
            dtype={
                "protein1": "category",
                "protein2": "category",
                "experiments": int,
                "combined_score": int,
            },
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = (
            self.__db().query("experiments != 0").query("combined_score >= 700")
        )  # threshold for high confidence https://string-db.org/help/faq/#how-to-extract-high-confidence-07-interactions-from-information-on-combined-score-in-proteinlinkstxtgz
        self.__interactions["subject"] = self.__interactions["protein1"].apply(
            lambda s: self.__NCBI.check_symbol(self.get_symbol_by_string(s))
        )  # checks protA symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["protein2"].apply(
            lambda s: self.__NCBI.check_symbol(self.get_symbol_by_string(s))
        )  # checks protB symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = "interacts_with"
        self.__interactions["subjectType"] = "protein"
        self.__interactions["objectType"] = "protein"
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()

    @property
    def symbol2string(self):
        return self.__symbol2string().copy()

    @property
    def symbol2string_asdict(self):
        return self.__symbol2string_asdict.copy()

    @property
    def string2symbol_asdict(self):
        return self.__string2symbol_asdict.copy()

    def get_string_by_symbol(self, symbol):
        return self.__symbol2string_asdict.get(symbol, f"{symbol} Not Found")

    def get_symbol_by_string(self, string):
        if not string.startswith("9606."):
            string = "9606." + string
        return self.__string2symbol_asdict.get(string, f"{string} Not Found")

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = [
                        f
                        for f in sources_data["STRING"]["files"]
                        if "protein.links" in f
                    ][0]
                    url = sources_data["STRING"]["files"][filename]["URL"]
                    self.v = re.findall("v(.+)\.txt", filename)[0]
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            import requests
            from bs4 import BeautifulSoup

            self.v = (
                BeautifulSoup(
                    requests.get("https://string-db.org/cgi/download").content,
                    "html5lib",
                )
                .find("ul")
                .findAll("li")[-1]
                .text
            )
            url = f"https://stringdb-static.org/download/protein.links.full.v{self.v}/9606.protein.links.full.v{self.v}.txt.gz"
            return url


# @profile()
class PRO(Database, metaclass=Singleton):
    """
    Protein Ontology by PRO Consortium reference class
    Contains info about proteins
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://proconsortium.org/download/documents/pro_license.txt",
            requirements=[NCBI, HGNC, GO],
        )
        self.__db = self._add_file(
            url="https://proconsortium.org/download/current/pro_reasoned.obo",
            version_marker="data-version",
        )

        PRO_base_categories = {
            "PR:000000001": "protein",
            "PR:000003507": "protein",
            "PR:000018263": "peptide",
            "PR:000018264": "peptide",
            "PR:000021935": "peptide",
            "PR:000021937": "peptide",
            "PR:000029067": "protein",
            "PR:000036194": "protein",
            "PR:000050098": "protein",
            "PR:000050567": "entityHavingProteicPart",
        }

        self.__proIsA = set()
        self.__pro2name = set()
        self.__pro2symbol = set()
        self.__external2symbol = set()
        self.__proRelationship = set()
        self.__pro2category = set()
        self.__pro2taxon = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            if hasattr(frame.id, "prefix") and frame.id.prefix in [
                "PR",
                "NCBIGene",
                "HGNC",
            ]:
                is_obsolete = False
                isA = []
                category = PRO_base_categories.get(str(frame.id))
                relationships = []
                synonyms = []
                taxon = None
                for clause in frame:
                    if isinstance(clause, fastobo.term.IsObsoleteClause):
                        is_obsolete = True
                        break
                    elif isinstance(clause, fastobo.term.NameClause):
                        name = clause.name
                    elif isinstance(
                        clause, fastobo.term.IsAClause
                    ) and clause.term.prefix in ["NCBIGene", "PR", "HGNC", "GO"]:
                        isA.append(clause.raw_value())
                    elif (
                        isinstance(clause, fastobo.term.SynonymClause)
                        and clause.synonym.scope == "EXACT"
                        and "PRO-short-label" in clause.raw_value()
                    ):
                        synonyms.append(clause.synonym.desc)
                    elif isinstance(clause, fastobo.term.CommentClause):
                        if (
                            not category
                            and "Category" in clause.comment
                            and "(was)" not in clause.comment
                        ):  # "(was)" used as filter in order to avoid errors related to obsolete terms
                            category = re.findall(
                                "Category=([a-z,A-Z,-]+)[\ \.]", clause.comment
                            )[0]
                    # elif isinstance(clause, fastobo.term.IntersectionOfClause) and clause.term.prefix in [
                    #     "CL",
                    #     "GO",
                    #     "HGNC",
                    #     "NCBIGene",
                    #     "NCBITaxon",
                    #     "PR",
                    # ]:
                    #     relationships.append((clause.raw_tag(), clause.raw_value()))
                    elif isinstance(clause, fastobo.term.RelationshipClause):
                        if clause.typedef.escaped == "only_in_taxon":
                            taxon = clause.term.local
                        elif clause.term.prefix in [
                            "CL",
                            "GO",
                            "HGNC",
                            "NCBIGene",
                            "NCBITaxon",
                            "PR",
                        ]:
                            relationships.append(
                                (clause.typedef.escaped, str(clause.term))
                            )
                    elif isinstance(
                        clause, fastobo.term.UnionOfClause
                    ) and clause.term.prefix in [
                        "CL",
                        "GO",
                        "HGNC",
                        "NCBIGene",
                        "NCBITaxon",
                        "PR",
                    ]:
                        relationships.append((clause.raw_tag(), clause.raw_value()))
                if not is_obsolete:
                    # if category:
                    self.__pro2category.add((str(frame.id), camelize(category)))
                    symbol = self.__NCBI.check_symbol(name.rstrip(" (human)"), synonyms)
                    if not symbol:
                        if frame.id.prefix == "NCBIGene":
                            symbol = self.__NCBI.get_symbol_by_id(frame.id.local)
                        elif frame.id.prefix == "HGNC":
                            symbol = self.__NCBI.check_symbol(
                                self.__HGNC.get_symbol_by_id(frame.id.local)
                            )
                    if category in {"gene", "organism-gene", "peptide"} and symbol:  # possibly a symbol has been retrieved from NCBI or HGNC
                        if "has_gene_template" not in {rel for rel, _ in relationships}:
                            relationships.append(
                                ("has_gene_template", symbol)
                            )  # in order to store mapping from PR to official symbols
                            self.__pro2name.add((symbol, self.__NCBI.get_name(symbol)))
                            self.__pro2category.add((symbol, "gene"))
                        self.__pro2symbol.add((str(frame.id), symbol))
                    elif category == "external" and symbol:
                        name = self.__NCBI.get_name(symbol)
                        self.__external2symbol.add(
                            (str(frame.id), symbol)
                        )  # used for mapping external identitiers to gene symbols
                        self.__pro2name.add((symbol, self.__NCBI.get_name(symbol)))
                        self.__pro2category.add((symbol, "gene"))
                    self.__proIsA.update([(str(frame.id), "is_a", id) for id in isA])
                    self.__proRelationship.update(
                        [(str(frame.id), *rel) for rel in relationships]
                    )
                    self.__pro2taxon.add((str(frame.id), taxon))
                    
                    if name:
                        self.__pro2name.add((str(frame.id), name))

        self.__ontology = pd.DataFrame(
            self.__proIsA, columns=["subject", "relation", "object"],
        )
        self.__ontology = pd.concat([
            self.__ontology,
            pd.DataFrame(
                self.__proRelationship, columns=["subject", "relation", "object"]
            )],
            ignore_index=True,
        )
        for role in ["subject", "object"]:
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__pro2taxon, columns=[role, f"{role}Taxon"]),
                on=role,
                how="left",
            ).reset_index(drop=True)

            self.__ontology = self.__ontology[
                self.__ontology[f"{role}Taxon"].isna()
                | self.__ontology[f"{role}Taxon"].str.fullmatch("9606|2759|33208")  # Homo sapiens, Eukaryota, Metazoa
            ][[col for col in self.__ontology.columns if "Taxon" not in col]]
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__pro2category, columns=[role, f"{role}Type"]),
                on=role,
                how="left",
            ).reset_index(drop=True)

            self.__ontology = (
                self.__ontology[
                    self.__ontology[f"{role}Type"].str.contains("[C,c]omplex")
                    | self.__ontology[f"{role}Type"].str.contains("[F,f]amily")
                    | self.__ontology[f"{role}Type"].str.contains("[G,g]ene")
                    | self.__ontology[f"{role}Type"].str.fullmatch("protein|peptide")
                    | self.__ontology[f"{role}Type"].str.fullmatch("external")
                    | self.__ontology[f"{role}Type"].str.contains("[M,m]odification")
                    | self.__ontology[f"{role}Type"].str.contains("[S,s]equence")
                    | self.__ontology[f"{role}Type"].str.fullmatch("seqgroup")
                    | self.__ontology[f"{role}Type"].str.fullmatch("entityHavingProteicPart")
                    | self.__ontology[f"{role}Type"].isna()
                ]
                .reset_index(drop=True)
                .replace(
                    {
                        "family": "proteinFamily",
                        "organismFamily": "organismProteinFamily",  # "proteinFamily",  # 
                        "gene": "protein",
                        "organismGene": "organismProtein",  # "protein",  # 
                        "complex": "proteinComplex",
                        "organismComplex": "organismComplex",  # "proteinComplex",
                        "organismModification": "organismProteinModification",  # "proteinModification",  # 
                        "modification": "proteinModification",
                        "organismSequence": "organismSequence",  # "sequence",  # 
                        "sequence": "sequence",
                        "seqgroup": "sequenceGroup",
                    }
                )
            )  # for the sake of clarity

            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__pro2name, columns=[role, f"{role}Name"]),
                on=role,
                how="left",
            ).reset_index(drop=True)

            self.__ontology[f"{role}Type"].replace(
                to_replace="external", value="protein", inplace=True
            )
            self.__ontology[role] = (
                self.__ontology[role]
                .map(dict(self.__external2symbol))
                .fillna(self.__ontology[role])
            )  # mapping external identifiers to known gene symbols
            self.__ontology = self.__ontology[
                ~(
                    self.__ontology[role].str.contains("HGNC:")
                    | self.__ontology[role].str.contains("NCBIGene:")
                )
            ].reset_index(
                drop=True
            )  # removes not matched HGNC and NCBIGene external terms

            self.__ontology[f"{role}Name"] = self.__ontology[f"{role}Name"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame(self.__GO.go2name, columns=[role, f"{role}Name"]),
                    on=role,
                    how="left",
                )[f"{role}Name"]
            )  # fills Name for GO objects

            self.__ontology[f"{role}Type"] = self.__ontology[f"{role}Type"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame(
                        self.__GO.go2namespace, columns=[role, f"{role}Type"]
                    ).replace("gene", "protein"),
                    on=role,
                    how="left",
                )[f"{role}Type"]
            )  # fills Type for GO objects

            # manually add CL name and type (in order to avoid a loop employing Uberon)
            self.__ontology[f"{role}Name"] = self.__ontology[f"{role}Name"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame({role: ["CL:0000236"], f"{role}Name": ["B cell"]}),
                    on=role,
                    how="left",
                )[f"{role}Name"]
            )  # fills Name for CL objects
            self.__ontology[f"{role}Type"] = self.__ontology[f"{role}Type"].fillna(
                self.__ontology[[role]].merge(
                    pd.DataFrame({role: ["CL:0000236"], f"{role}Type": ["cell"]}),
                    on=role,
                    how="left",
                )[f"{role}Type"]
            )  # fills Type for CL objects

        self.__ontology["source"] = ["PRO"] * len(self.__ontology)
        self.__ontology = self.__ontology[
                ~(self.__ontology.subjectType.str.startswith("organism") + self.__ontology.objectType.str.startswith("organism"))
            ]  # not consider organism specific terms
        self.__ontology = self.__ontology.drop_duplicates(ignore_index=True).astype(
            {
                "subject": "string",
                "relation": "category",
                "object": "string",
                "subjectName": "string",
                "objectName": "string",
                "subjectType": "category",
                "objectType": "category",
                "source": "string",
            }
        )

        # dicts for info retrieval
        self.__pro2name_asdict = dict(self.__pro2name)
        self.__pro2category_asdict = dict(self.__pro2category)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def pro2name(self):
        return self.__pro2name.copy()

    @property
    def pro2category(self):
        return self.__pro2category.copy()

    @property
    def pro2symbol(self):
        return self.__pro2symbol.copy()

    @property
    def external2symbol(self):
        return self.__external2symbol.copy()

    @property
    def database(self):
        return self.__db().copy()

    @property
    def ontology(self):
        return self.__ontology.copy()

    def get_name(self, id):
        """Returns the name of a PRO id"""
        if not id.startswith("PR:"):
            id = f"PR:{id}"
        return self.__pro2name_asdict.get(id, f"Id {id} not found")

    def get_category(self, id):
        """Returns the category associated to a PRO id"""
        if not id.startswith("PR:"):
            id = f"PR:{id}"
        return self.__pro2category_asdict.get(id, f"Id {id} not found")


# @profile()
class DrugBank(Database, metaclass=Singleton):
    """
    DrugBank reference class
    Contains info about drugs
    """

    def __init__(self, update=None):
        try:
            email = os.environ["DRUGBANK_EMAIL"]
            password = os.environ["DRUGBANK_PASSWORD"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                email = credentials["DRUGBANK_EMAIL"]
                password = credentials["DRUGBANK_PASSWORD"]
            except:
                raise RuntimeError(
                    "No DrugBank credentials found, "
                    "drugs information will not be collected"
                )
        session = requests.Session()
        import base64

        session.headers.update(
            {
                "Authorization": f"Basic {base64.b64encode(f'{email}:{password}'.encode('ascii')).decode('ascii')}"
            }
        )
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC 4.0",
            license_url="https://go.drugbank.com/releases/latest#full",
            registration_required=True,
            session=session,
            requirements=[NCBI, PathwayCommons],
        )

        def read_drugbank_full_database(filepath):
            # preparing namedtuples
            from collections import namedtuple

            atc_code = namedtuple(
                typename="atc_code",
                field_names=["level5", "level4", "level3", "level2", "level1"],
            )
            atc_code_level = namedtuple(
                typename="atc_code_level", field_names=["code", "name"]
            )
            category = namedtuple(typename="category",
                                  field_names=["name", "mesh_id"])
            interacting_drug = namedtuple(
                typename="interacting_drug",
                field_names=["name", "drugbank_id", "description"],
            )
            experimental_property = namedtuple(
                typename="experimental_property",
                field_names=["name", "value", "source"],
            )
            calculated_property = namedtuple(
                typename="calculated_property", field_names=["name", "value", "source"]
            )
            external_identifiers = namedtuple(
                typename="external_identifiers", field_names=["resource", "id"]
            )
            pathway = namedtuple(
                typename="pathway", field_names=["name", "smpdb_id", "category"]
            )
            protein_function = namedtuple(
                typename="protein_function", field_names=["general", "specific"]
            )
            protein = namedtuple(
                typename="protein",
                field_names=[
                    "drug_actions",
                    "cellular_location",
                    "chromosome_location",
                    "function",
                    "id",
                    "name",
                    "organism",
                    "swiss_prot_id",
                    "symbol",
                    "synonyms",
                    "type",
                ],
            )
            biological_entity = namedtuple(
                typename="biological_entity",
                field_names=["drug_actions", "id", "name", "organism", "type"],
            )
            small_molecule = namedtuple(
                typename="small_molecule",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "calculated_properties",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "first_on_market",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "products",
                    "synonyms",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            biotech = namedtuple(
                typename="biotech",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "products",
                    "synonyms",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            # parsing database
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open(z.filelist[0].filename) as f:
                    from lxml import objectify

                    drugbank_database = objectify.parse(f).getroot().drug
            drugs_namedtuple = namedtuple(
                typename="drugs",
                field_names=tuple(d["drugbank-id"] for d in drugbank_database),
            )
            drugs = drugs_namedtuple(
                *[
                    small_molecule(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[
                                                     0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            calculated_property(
                                str(prop.kind), str(
                                    prop.value), str(prop.source)
                            )
                            for prop in d["calculated-properties"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(
                                    prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(
                                str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        (lambda l: str(min(l)) if len(l) else "")(
                            {product["started-marketing-on"] for product in d.products.getchildren() if product["started-marketing-on"] != ""}),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(
                                p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple({str(product.name) for product in d.products.getchildren()}.union(
                            {str(intb.name) for intb in d["international-brands"].getchildren()})),
                        tuple(str(synonym)
                              for synonym in d.synonyms.getchildren()),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "small_molecule",
                    )
                    if d.values()[0] == "small molecule"
                    else biotech(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[
                                                     0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(
                                    prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(
                                str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(
                                p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple({str(product.name) for product in d.products.getchildren()}.union(
                            {str(intb.name) for intb in d["international-brands"].getchildren()})),
                        tuple(str(synonym)
                              for synonym in d.synonyms.getchildren()),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "biotech",
                    )
                    for d in tqdm(
                        drugbank_database, desc="Collecting Drugs Data from DrugBank"
                    )
                ]
            )

            return drugs

        self.__db = self._add_file(
            url=self.get_current_release_url(),
            custom_read_function=read_drugbank_full_database,
            retrieved_version=self.v,  # retrieved by get_current_release_url
        )

        self.__name2id = {drug.name: drug.id for drug in self.__db()}
        self.__drugIDs = {drug.id for drug in self.__db()}
        self.__drugNames = {drug.name for drug in self.__db()}

        self.__inchiKeyBase2name = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.name
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2name[None]]
        self.__inchiKeyBase2id = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.id
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2id[None]]

        self.__drug2target = pd.DataFrame(
            (
                (drug.id, target.symbol, drug.name, target.name)
                for drug in self.drugs
                for target in drug.targets
                if target.type == "protein"
                and target.organism == "Humans"
                and target.symbol != None
            ),
            columns=["drugId", "geneSymbol", "drugName", "geneName"],
            dtype="string",
        )
        self.__drug2target = pd.merge(self.__drug2target, self._NCBI.gene2name, on="geneSymbol", how="left")
        self.__drug2target["geneName"] = self.__drug2target.geneName_y.fillna(self.__drug2target.geneName_x)
        self.__drug2target = self.__drug2target[["drugId", "geneSymbol", "drugName", "geneName"]].astype("string")

        self.__drug2transporter = pd.DataFrame(
            (
                (drug.id, carrier.symbol, drug.name, carrier.name)
                for drug in self.drugs
                for carrier in drug.carriers
                if carrier.type == "protein"
                and carrier.organism == "Humans"
                and carrier.symbol != None
            ),
            columns=["drugId", "geneSymbol", "drugName", "geneName"],
            dtype="string",
        )
        self.__drug2transporter = pd.merge(self.__drug2transporter, self._NCBI.gene2name, on="geneSymbol", how="left")
        self.__drug2transporter["geneName"] = self.__drug2transporter.geneName_y.fillna(self.__drug2transporter.geneName_x)
        self.__drug2transporter = self.__drug2transporter[["drugId", "geneSymbol", "drugName", "geneName"]].astype("string")

        self.__drug2enzyme = pd.DataFrame(
            (
                (drug.id, enzyme.symbol, drug.name, enzyme.name)
                for drug in self.drugs
                for enzyme in drug.enzymes
                if enzyme.type == "protein"
                and enzyme.organism == "Humans"
                and enzyme.symbol != None
            ),
            columns=["drugId", "geneSymbol", "drugName", "geneName"],
            dtype="string",
        )
        self.__drug2enzyme = pd.merge(self.__drug2enzyme, self._NCBI.gene2name, on="geneSymbol", how="left")
        self.__drug2enzyme["geneName"] = self.__drug2enzyme.geneName_y.fillna(self.__drug2enzyme.geneName_x)
        self.__drug2enzyme = self.__drug2enzyme[["drugId", "geneSymbol", "drugName", "geneName"]].astype("string")

        self.__drug2gene = pd.concat([self.__drug2target, self.__drug2transporter, self.__drug2enzyme])

        self.__drug2pathway = pd.DataFrame(
            (
                (drug.id, pathway.smpdb_id, drug.name, pathway.name, pathway.category)
                for drug in self.drugs
                for pathway in drug.pathways
            ),
            columns=["drugId", "pathwayId", "drugName", "pathwayName", "pathwayCategory"],
        )
        self.__drug2pathway = pd.merge(self.__drug2pathway, self.__PathwayCommons.pathways, on="pathwayId", how="left")
        self.__drug2pathway["pathwayName"] = self.__drug2pathway.pathwayName_y.fillna(self.__drug2pathway.pathwayName_x)
        self.__drug2pathway = self.__drug2pathway[
            ["drugId", "pathwayId", "drugName", "pathwayName", "pathwayCategory"]
        ].astype({"drugId":"string", "pathwayId":"string", "drugName":"string", "pathwayName":"string", "pathwayCategory":"category"})

        self.__drug_interactions = pd.DataFrame(
            (
                (drug.id, interaction.drugbank_id, drug.name, interaction.name)
                for drug in self.drugs
                for interaction in drug.drug_interactions
                if interaction.drugbank_id != None
            ),
            columns=["source", "target", "sourceName", "targetName"],
            dtype="string",
        )

        drug2target = self.__drug2target.rename(columns={"drugId":"subject", "geneSymbol":"object", "drugName":"subjectName", "geneName":"objectName"})
        drug2target["subjectType"] = "drugId"
        drug2target["objectType"] = "geneSymbol"
        drug2target["relation"] = "targets"
        drug2target["source"] = "DrugBank"

        drug2transporter = self.__drug2transporter.rename(columns={"drugId":"subject", "geneSymbol":"object", "drugName":"subjectName", "geneName":"objectName"})
        drug2transporter["subjectType"] = "drugId"
        drug2transporter["objectType"] = "geneSymbol"
        drug2transporter["relation"] = "transported_by"
        drug2transporter["source"] = "DrugBank"

        drug2enzyme = self.__drug2enzyme.rename(columns={"drugId":"subject", "geneSymbol":"object", "drugName":"subjectName", "geneName":"objectName"})
        drug2enzyme["subjectType"] = "drugId"
        drug2enzyme["objectType"] = "geneSymbol"
        drug2enzyme["relation"] = "metabolized_by"
        drug2enzyme["source"] = "DrugBank"

        drug2protein = pd.concat([drug2target, drug2transporter, drug2enzyme])

        drug2pathway = self.__drug2pathway
        drug2pathway.drop("pathwayCategory", axis=1, inplace=True)
        drug2pathway["pathwayId"] = "PathBank:" + drug2pathway["pathwayId"]
        drug2pathway = drug2pathway[drug2pathway.pathwayId.isin(self.__PathwayCommons.pathways.pathwayId)].rename(columns={"drugId":"subject", "pathwayId":"object", "drugName":"subjectName", "pathwayName":"objectName"})
        drug2pathway["subjectType"] = "drugId"
        drug2pathway["objectType"] = "pathwayId"
        drug2pathway["relation"] = "involved_in"
        drug2pathway["source"] = "DrugBank"

        drug_interactions = self.__drug_interactions
        # keep only one direction of edges
        drug_interactions.loc[:] = np.take_along_axis(drug_interactions.values, np.tile(np.argsort(drug_interactions[["source", "target"]], axis=1), (1,2)) + np.array([0,0,2,2]), axis=1)
        drug_interactions.drop_duplicates(inplace=True, ignore_index=True)
        drug_interactions = drug_interactions.rename(columns={"source":"subject", "target":"object", "sourceName":"subjectName", "targetName":"objectName"})
        drug_interactions["subjectType"] = "drugId"
        drug_interactions["objectType"] = "drugId"
        drug_interactions["relation"] = "interacts_with"
        drug_interactions["source"] = "DrugBank"

        # collect all data in a single dataframe
        self.__database = pd.concat([drug2protein, drug2pathway, drug_interactions])
        self.__database = (
            self.__database[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__database.copy()

    @property
    def drugs(self):
        return self.__db()
    
    @property
    def drug2target(self):
        return self.__drug2target.copy()
    
    @property
    def drug2enzyme(self):
        return self.__drug2enzyme.copy()
    
    @property
    def drug2transporter(self):
        return self.__drug2transporter.copy()
    
    @property
    def drug2gene(self):
        return self.__drug2gene.copy()
    
    @property
    def drug2pathway(self):
        return self.__drug2pathway.copy()
    
    @property
    def drug_interactions(self):
        return self.__drug_interactions.copy()

    @property
    def inchiKeyBase2id(self):
        return self.__inchiKeyBase2id

    def get_id_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2id.get(inchiKeyBase)

    @property
    def inchiKeyBase2name(self):
        return self.__inchiKeyBase2name

    def get_name_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2name.get(inchiKeyBase)

    def get(self, query):
        """
            Returns a namedtuple with relevant data about the requested drug

            Accepts DrugBank IDs or drug names (returns only the exact matches)
        """
        if not isinstance(query, str):
            return None
        else:
            if query.startswith("DB") and query in self.__drugIDs:
                return getattr(self.__db(), query)
            elif query in self.__drugNames:
                return getattr(self.__db(), self.__name2id[query])
            else:
                return None

    def search(self, query):
        """
            Returns a namedtuple with relevant data about the requested drug

            Accepts DrugBank IDs or drug names (returns the best match, if relevant)
        """
        if not isinstance(query, str):
            return None
        else:
            if query.startswith("DB") and query in self.__drugIDs:
                return getattr(self.__db(), query)
            elif query in self.__drugNames:
                return getattr(self.__db(), self.__name2id[query])
            else:
                best_match = get_best_match(query, self.__drugNames)
                if best_match:
                    return getattr(self.__db(), self.__name2id[best_match])
                else:
                    return None

    def search_drug(self, query):
        """
            Alias for search
        """
        return self.search(query)

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = list(
                        sources_data["DrugBank"]["files"].keys())[0]
                    url = sources_data["DrugBank"]["files"][filename]["URL"]
                    version = sources_data["DrugBank"]["files"][filename]["version"]
                    self.v = re.findall("(.+)\    ", version)[
                        0
                    ]  # before four spaces (tab)
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            from bs4 import BeautifulSoup

            self.v = re.findall(
                "Version ([0-9]+\.[0-9]+\.[0-9]+) ",
                BeautifulSoup(
                    requests.get(
                        "https://go.drugbank.com/releases/latest").content,
                    "html5lib",
                ).head.title.text,
            )[0]
            url = f"https://go.drugbank.com/releases/{self.v.replace('.', '-')}/downloads/all-full-database"
            if os.path.isfile("data/sources/sources.json"):
                try:
                    import json

                    with open("data/sources/sources.json", "r+") as infofile:
                        sources_data = json.load(infofile)
                        local_url = sources_data["DrugBank"]["files"][
                            list(sources_data["DrugBank"]["files"].keys())[0]
                        ]["URL"]
                    if (
                        local_url == url
                    ):  # if there is not a newer version online it doesn't update the database
                        self._update = False
                except:
                    pass
            return url


# @profile()
class DrugCentral(Database, metaclass=Singleton):
    """
    DrugCentral reference class
    Contains info about drugs
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY-SA 4.0",
            license_url="https://drugcentral.org/privacy",
            requirements=[MONDO, DisGeNET, DrugBank],
        )

        self.__sql = SQLdatabase(host="unmtid-dbs.net", database="drugcentral", port=5433, user="drugman", password="dosage")

        self.v, database_date = self.__sql.query(
            """
            select *
            from dbversion
            """
        ).iloc[0]

        self.__db = self._add_file(
            url="jdbc:postgresql://unmtid-dbs.net:5433/drugcentral",
            retrieved_version=self.v,
            retrieved_filename="drug2disease.tsv",
            retrieved_date=database_date.to_pydatetime().astimezone(timezone.utc),
            retrieved_content=self.__sql.query(
                """
                select identifier.identifier , relationships.struct_id, relationships.relationship_name, relationships.umls_cui
                from omop_relationship relationships
                    join identifier on identifier.struct_id = relationships.struct_id
                        where identifier.id_type = 'DRUGBANK_ID'
                            and identifier.parent_match is null
                """
            )
        )

        self.__drug2disease = pd.merge(
            self.__db(),
            pd.DataFrame(self.__MONDO.umls2mondo, columns=["umls_cui", "diseaseId"]),
            on="umls_cui",
            how="left").dropna(ignore_index=True)[["identifier",
                                  "relationship_name",
                                  "diseaseId",
                                  "source"]].rename(columns={"identifier":"drugId",
                                                              "relationship_name":"relation"})
        self.__drug2disease = self.__drug2disease.merge(
                pd.DataFrame(
                    ((drug.id, drug.name) for drug in self.__DrugBank.drugs),
                    columns=["drugId", "subjectName"]
                ),
                on="drugId",
                how="left",
            )
        self.__drug2disease = self.__drug2disease.merge(
            pd.DataFrame(self.__MONDO.mondo2name,
                 columns=["diseaseId", "diseaseName"],
                ),
            on="diseaseId",
            how="left",
        )
        self.__drug2disease = self.__drug2disease.rename(columns={"drugId":"subject",
                                                                "diseaseId":"object",
                                                                "diseaseName":"objectName"})
        self.__drug2disease["subjectType"] = "drug"
        self.__drug2disease["objectType"] = "disease"
        self.__drug2disease = self.__drug2disease[
            [
                "subject",
                "relation",
                "object",
                "subjectName",
                "objectName",
                "subjectType",
                "objectType",
                "source",
            ]
        ].dropna(ignore_index=True)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()
    
    @property
    def drug2disease(self):
        return self.__drug2disease.copy()
    
    def query(self, query):
        return self.__sql.query(query)