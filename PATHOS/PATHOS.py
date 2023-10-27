import pandas as pd
import numpy as np

from utils import integrate_dataframes, profile, Singleton
import databases as dbs

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "PATHOS" if __name__ == "__main__" else __name__
log = logging.getLogger(logging_name)


# @profile()
class PPI(metaclass=Singleton):
    """
    Collect and merge protein-protein interactions for Homo sapiens
    Sources are APID, BioGRID, HuRI, InnateDB, INstruct, IntAct, SignLink, STRING
    """

    def __init__(self, update=True):
        log = logging.getLogger("PPI")
        log.info("Collecting protein-protein interactions")
        self.__update = update
        log.info(
            f"Retrieving source databases of protein-protein interactions for {self.__class__.__name__}"
        )
        self.__sources = {
            db.name: db
            for db in [
                dbs.APID(update=self.__update),
                dbs.BioGRID(update=self.__update),
                dbs.HPRD(update=self.__update),
                dbs.HuRI(update=self.__update),
                dbs.InnateDB(update=self.__update),
                dbs.INstruct(update=self.__update),
                dbs.IntAct(update=self.__update),
                dbs.PINA(update=self.__update),
                dbs.SignaLink(update=self.__update),
                dbs.STRING(update=self.__update),
            ]
        }
        self.__versions = {db.name: db.version for db in self.__sources.values()}

        # save files only if there are updates in the databases
        if any([db.updated for db in self.__sources.values()]):
            log.info(f"Integrating source databases for {self.__class__.__name__}")
            self.__interactome = integrate_dataframes(
                [db.interactions for db in self.__sources.values()],
                columns_to_join=["source"],
            )
            self.__interactome.to_csv(
                "data/interactome.tsv.gz", sep="\t", compression="gzip",
            )
            self.__interactome[["subject", "object"]].rename(
                columns={"subject": "source", "object": "target"}
            ).to_csv(
                "data/interactome_slim.tsv.gz",
                sep="\t",
                compression="gzip",
                index=False,
            )
        else:
            self.__interactome = pd.read_csv(
                "data/interactome.tsv.gz",
                sep="\t",
                compression="gzip",
                index_col=0,
                dtype={
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                },
            )

        self.__proteins = pd.concat(
            [
                self.__interactome[["subject", "subjectName"]].rename(
                    columns={"subject": "proteinSymbol", "subjectName": "proteinName"}
                ),
                self.__interactome[["object", "objectName"]].rename(
                    columns={"object": "proteinSymbol", "objectName": "proteinName"}
                ),
            ]
        ).drop_duplicates(ignore_index=True)

        self.__interactome_undirected = self.__interactome.copy()
        self.__interactome_undirected.loc[:, ["subject", "object", "subjectName", "objectName", "subjectType", "objectType"]] = np.take_along_axis(self.__interactome_undirected[["subject", "object", "subjectName", "objectName", "subjectType", "objectType"]].values, np.tile(np.argsort(self.__interactome_undirected[["subject", "object"]], axis=1), (1,3)) + np.array([0,0,2,2,4,4]), axis=1)
        self.__interactome_undirected.drop_duplicates(inplace=True, ignore_index=True)


        log.info(f"{self.__class__.__name__} ready!")

    @property
    def interactome(self):
        return self.__interactome.copy()
    
    @property
    def interactome_undirected(self):
        return self.__interactome_undirected.copy()

    @property
    def proteins(self):
        return self.__proteins.copy()

    @property
    def sources(self):
        return self.__sources.copy()

    @property
    def version(self):
        print(f"{self.__class__.__name__} sources versions:")
        for key, value in self.__versions.items():
            print(f"\t{key}:\t{value}")

    @property
    def versions(self):
        return self.__versions

    def __call__(self):
        return self.__interactome.copy()

    def __repr__(self):
        return "Human Protein-Protein Interactome"

    def __str__(self):
        return f"Human Protein-Protein Interactome\nTotal Interactions: {len(self.__interactome)}\nTotal Proteins: {len(self.__proteins)}"


@profile()
class PATHOS(metaclass=Singleton):
    def __init__(self, update=True):
        log = logging.getLogger("PATHOS")
        log.info("Building the Knowledge Graph")
        self.__update = update
        self.__ppi = PPI(update=self.__update)
        self.__sources = self.__ppi.sources
        self.__sources.update(
            {
                db.name: db
                for db in [
                    dbs.NCBI(update=self.__update),
                    dbs.DisGeNET(update=self.__update),
                    dbs.MONDO(update=self.__update),
                    dbs.HPO(update=self.__update),
                    dbs.DISEASES(update=self.__update),
                    dbs.PathwayCommons(update=self.__update),
                    dbs.Bgee(update=self.__update),
                    dbs.Uberon(update=self.__update),
                    dbs.GO(update=self.__update),
                    dbs.PRO(update=self.__update),
                    dbs.DrugBank(update=self.__update),
                    dbs.DrugCentral(update=self.__update),
                ]
            }
        )
        if any([db.updated for db in self.__sources.values()]):
            # proteins
            self.__interactome = self.__ppi.interactome_undirected

            # Collect info about diseases
            # and the relationships among themselves and with genes.
            # Sources are MONDO, HPO, DisGeNET, DISEASES
            log = logging.getLogger("PATHOS:diseases")
            log.info("Collecting info about diseases")
            self.__disease2phenotype = self.__sources["HPO"].disease2phenotype
            self.__disease2phenotype = self.__disease2phenotype.rename(
                columns={
                    "diseaseId": "subject",
                    "phenotypeId": "object",
                    "diseaseName": "subjectName",
                    "phenotypeName": "objectName",
                }
            )
            self.__disease2phenotype["subjectType"] = "disease"
            self.__disease2phenotype["objectType"] = "phenotype"
            self.__disease2phenotype["relation"] = "has_phenotype"
            self.__disease2phenotype = self.__disease2phenotype[
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
            ].reset_index(drop=True)
            self.__protein2disease = (
                self.__sources["DisGeNET"]
                .database.query("EI == 1")[["geneSymbol", "diseaseId", "source"]]
                .rename(columns={"diseaseId": "diseaseId_UMLS"})
            )
            self.__protein2disease = self.__protein2disease.merge(
                pd.DataFrame(
                    self.__sources["MONDO"].umls2mondo,
                    columns=["diseaseId_UMLS", "diseaseId"],
                ),
                on="diseaseId_UMLS",
                how="left",
            )[["geneSymbol", "diseaseId", "source"]]
            self.__protein2disease = integrate_dataframes(
                [self.__protein2disease, self.__sources["DISEASES"].database],
                common_columns=["geneSymbol", "diseaseId"],
            )
            self.__protein2disease = (
                self.__protein2disease.merge(
                    self.__sources["NCBI"].gene2name, on="geneSymbol"
                )
                .dropna(subset=["geneName"])
                .reset_index(drop=True)
            )
            self.__protein2disease.dropna(
                subset=["geneSymbol", "geneName"], inplace=True
            )
            self.__protein2disease = (
                self.__protein2disease.merge(
                    pd.DataFrame(
                        self.__sources["MONDO"].mondo2name,
                        columns=["diseaseId", "diseaseName"],
                    ),
                    on="diseaseId",
                )
                .sort_values(by="geneSymbol")
                .reset_index(drop=True)
            )
            self.__protein2disease = self.__protein2disease[
                ["geneSymbol", "diseaseId", "geneName", "diseaseName", "source"]
            ]
            # keep local (only needed for building the collection)
            protein2disease = self.__protein2disease.rename(
                columns={
                    "geneSymbol": "subject",
                    "diseaseId": "object",
                    "geneName": "subjectName",
                    "diseaseName": "objectName",
                }
            )
            protein2disease["subjectType"] = "protein"
            protein2disease["objectType"] = "disease"
            protein2disease["relation"] = "related_to_disease"  # to check
            protein2disease = protein2disease[
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
            ].reset_index(drop=True)
            self.__diseases = self.__sources["MONDO"].diseases

            # Collect info about pathways
            # and the relationships among themselves and with genes.
            # Source is PathwayCommons
            log = logging.getLogger("PATHOS:pathways")
            log.info("Collecting info about pathways")
            self.__protein2pathway = self.__sources["PathwayCommons"].gene2pathway
            self.__pathways = self.__sources["PathwayCommons"].pathways

            # Collect info about gene expression
            # Source is Bgee
            log = logging.getLogger("PATHOS:anatomicalEntities")
            log.info("Collecting info about anatomicalEntities")
            self.__protein_expression = self.__sources["Bgee"].gene_expression
            protein_expression = self.__protein_expression
            protein_expression["subjectType"] = "protein"
            protein_expression["relation"] = "expressed_in"  # to check
            protein_expression = protein_expression.rename(
                columns={
                    "geneSymbol": "subject",
                    "anatomicalEntityId": "object",
                    "geneName": "subjectName",
                    "anatomicalEntityName": "objectName",
                    "anatomicalEntityType": "objectType",
                }
            )
            protein_expression = protein_expression[
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

            # Collect info about biological processes,
            # molecular functions,
            # and cellular components
            # Source is Gene Ontology
            log = logging.getLogger("PATHOS:GO")
            log.info("Collecting info from GO")
            self.__protein2go = self.__sources["GO"].gene2go
            self.__biologicalProcesses = self.__sources["GO"].biologicalProcesses
            self.__cellularComponents = self.__sources["GO"].cellularComponents
            self.__molecularFunctions = self.__sources["GO"].molecularFunctions

            # Collect info about drugs
            log = logging.getLogger("PATHOS:DrugBank")
            log.info("Collecting info from DrugBank")
            drug_relations = self.__sources["DrugBank"].database
            drug_relations.subjectType = drug_relations.subjectType.replace({"drugId":"drug", "diseaseId":"disease", "pathwayId": "pathway,", "geneSymbol":"protein"})
            drug_relations.objectType = drug_relations.objectType.replace({"drugId":"drug", "diseaseId":"disease", "pathwayId": "pathway", "geneSymbol":"protein"})
            self.__drug2target = self.__sources["DrugBank"].drug2target
            self.__drug2transporter = self.__sources["DrugBank"].drug2transporter
            self.__drug2enzyme = self.__sources["DrugBank"].drug2enzyme
            self.__drug2protein = self.__sources["DrugBank"].drug2gene
            self.__drug2pathway = self.__sources["DrugBank"].drug2pathway
            self.__drug_interactions = self.__sources["DrugBank"].drug_interactions
            
            self.__drug2disease = self.__sources["DrugCentral"].drug2disease

            log = logging.getLogger("PATHOS")
            log.info("Integrating Dataframes...")
            self.__collection = integrate_dataframes(
                [
                    self.__interactome,
                    self.__sources["PRO"].ontology.replace(
                        "has_gene_template", "has_symbol"
                    ),
                    self.__sources["MONDO"].ontology,
                    protein2disease,
                    self.__disease2phenotype,
                    self.__sources["PathwayCommons"].database,
                    protein_expression,
                    self.__sources["Uberon"].ontology,
                    self.__sources["GO"].ontology,
                    drug_relations,
                    self.__drug2disease,
                ]
            )
            self.__collection.sort_values(
                by=["subjectType", "objectType", "subject", "relation", "object"],
                ignore_index=True,
                inplace=True,
            )  # it could be useless to sort it multiple times
            self.__collection_with_negatives = self.__collection.copy()
            self.__collection = self.__collection[
                ~self.__collection["relation"].str.startswith("NOT|")
            ]
            self.__collection.sort_values(
                by=["subject", "relation", "object", "subjectType", "objectType"],
                ignore_index=True,
                inplace=True,
            )

            self.__collection = self.collection.astype(
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

            self.__collection.to_csv(
                "data/KG.tsv.gz", sep="\t", compression="gzip",
            )
            self.__collection[["subject", "relation", "object"]].to_csv(
                "data/KG_slim.tsv.gz", sep="\t", compression="gzip", index=False,
            )
        else:
            self.__collection = pd.read_csv(
                "data/KG.tsv.gz",
                sep="\t",
                compression="gzip",
                index_col=0,
                dtype={
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                },
            )

    @property
    def ppi(self):
        return self.__ppi
    
    @property
    def interactome(self):
        return self.__ppi

    @property
    def disease2phenotype(self):
        return self.__disease2phenotype

    @property
    def protein2disease(self):
        return self.__protein2disease

    @property
    def diseases(self):
        return self.__diseases

    @property
    def protein2pathway(self):
        return self.__protein2pathway

    @property
    def pathways(self):
        return self.__pathways

    @property
    def protein_expression(self):
        return self.__protein_expression

    @property
    def protein2go(self):
        return self.__protein2go

    @property
    def biologicalProcesses(self):
        return self.__biologicalProcesses

    @property
    def cellularComponents(self):
        return self.__cellularComponents

    @property
    def molecularFunctions(self):
        return self.__molecularFunctions
    
    @property
    def drug2target(self):
        return self.__drug2target
    
    @property
    def drug2enzyme(self):
        return self.__drug2enzyme
    
    @property
    def drug2transporter(self):
        return self.__drug2transporter
    
    @property
    def drug2protein(self):
        return self.__drug2protein
    
    @property
    def drug2pathway(self):
        return self.__drug2pathway
    
    @property
    def drug_interactions(self):
        return self.__drug_interactions

    @property
    def collection(self):
        return self.__collection

    @property
    def collection_with_negatives(self):
        return self.__collection_with_negatives

    @property
    def sources(self):
        return self.__sources

    @property
    def version(self):
        print(f"{self.__class__.__name__} sources versions:")
        for key, value in self.__versions.items():
            print(f"\t{key}:\t{value}")

    @property
    def versions(self):
        return self.__versions

    def __call__(self):
        return self.__collection

    def __repr__(self):
        return "PATHOS (Knowledge Graph about PATHologies of HOmo Sapiens)"

    def __str__(self):
        return "PATHOS (Knowledge Graph about PATHologies of HOmo Sapiens)"


###     --- EXECUTION ---       ###

if __name__ == "__main__":
    PATHOS = PATHOS()
