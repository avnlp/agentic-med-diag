"""Medical GraphRAG knowledge-graph schema: 13 entity types, 25 relation types."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


# ── 13 extractable entity types ──────────────────────────────────────────────

ENTITY_TYPES: list[EntityType] = [
    EntityType(
        label="Disease",
        gliner_label="disease, disorder, syndrome, condition, or neoplasm",
        description=(
            "Named disease, medical condition, disorder, syndrome, or neoplasm. "
            "Examples: 'type 2 diabetes mellitus', 'myocardial infarction', "
            "'lung adenocarcinoma'. Use lowercase canonical names."
        ),
        key_property="name",
    ),
    EntityType(
        label="Drug",
        gliner_label="drug, medication, or pharmaceutical agent",
        description=(
            "Specific pharmaceutical drug or therapeutic agent. "
            "Use generic/INN names. Examples: 'metformin', 'lisinopril', "
            "'oseltamivir', 'adalimumab'."
        ),
        key_property="genericName",
    ),
    EntityType(
        label="DrugClass",
        gliner_label="pharmacological drug class or drug category",
        description=(
            "Category of drugs sharing mechanism or structure. "
            "Examples: 'ACE inhibitors', 'beta-blockers', "
            "'SSRIs', 'fluoroquinolones'."
        ),
        key_property="name",
    ),
    EntityType(
        label="Symptom",
        gliner_label="symptom, clinical sign, or clinical finding",
        description=(
            "Clinical symptom, sign, laboratory finding, or imaging finding. "
            "Examples: 'dyspnea', 'S3 gallop', "
            "'elevated troponin', 'fever'."
        ),
        key_property="name",
    ),
    EntityType(
        label="Pathogen",
        gliner_label="infectious pathogen, bacterium, virus, fungus, or parasite",
        description=(
            "Infectious organism. Examples: 'Streptococcus pneumoniae', "
            "'SARS-CoV-2', 'Candida albicans', 'Plasmodium falciparum'."
        ),
        key_property="name",
    ),
    EntityType(
        label="AnatomicalStructure",
        gliner_label="anatomical structure, body part, organ, vessel, or tissue",
        description=(
            "Anatomical body part, organ, tissue, vessel, nerve, or cell type. "
            "Examples: 'left ventricle', 'femoral artery', "
            "'hippocampus', 'hepatocyte'."
        ),
        key_property="name",
    ),
    EntityType(
        label="Procedure",
        gliner_label="surgical procedure or therapeutic intervention",
        description=(
            "Surgical or interventional therapeutic procedure. "
            "Examples: 'coronary artery bypass graft', "
            "'appendectomy', 'hemodialysis'."
        ),
        key_property="name",
    ),
    EntityType(
        label="DiagnosticTest",
        gliner_label="diagnostic test, laboratory test, or imaging study",
        description=(
            "Lab test, imaging study, or diagnostic measure. "
            "Examples: 'HbA1c', 'echocardiogram', "
            "'serum creatinine', 'CT angiography'."
        ),
        key_property="name",
    ),
    EntityType(
        label="RiskFactor",
        gliner_label="disease risk factor or predisposing condition",
        description=(
            "Factor that predisposes to disease. "
            "Examples: 'hypertension', 'smoking', "
            "'obesity', 'family history of MI'."
        ),
        key_property="name",
    ),
    EntityType(
        label="Gene",
        gliner_label="human gene or genetic locus",
        description=(
            "Human gene by HGNC symbol. "
            "Examples: 'CFTR', 'BRCA1', "
            "'TP53', 'HER2', 'EGFR', 'KRAS'."
        ),
        key_property="symbol",
    ),
    EntityType(
        label="Protein",
        gliner_label="protein, enzyme, receptor, transporter, or molecular target",
        description=(
            "Protein, enzyme, receptor, channel, or transporter. "
            "Examples: 'CYP3A4', 'ACE', 'neuraminidase', "
            "'insulin receptor'."
        ),
        key_property="name",
    ),
    EntityType(
        label="Pathway",
        gliner_label="biochemical pathway or signaling pathway",
        description=(
            "Biochemical or signaling pathway. "
            "Examples: 'glycolysis', "
            "'JAK-STAT signaling', 'mTOR pathway'."
        ),
        key_property="name",
    ),
    EntityType(
        label="MechanismOfAction",
        gliner_label="mechanism of action or pathophysiological mechanism",
        description=(
            "Drug mechanism or disease pathomechanism. "
            "Examples: 'neuraminidase inhibition', "
            "'calcium channel blockade'."
        ),
        key_property="name",
    ),
]

# ── 25 relation types ─────────────────────────────────────────────────────────

RELATION_TYPES: list[RelationType] = [
    RelationType(
        type="HAS_SYMPTOM",
        glirel_label="disease has symptom or clinical manifestation",
        head_labels=["Disease"],
        tail_labels=["Symptom"],
        properties=["frequency", "pathognomonic"],
    ),
    RelationType(
        type="TREATED_BY",
        glirel_label="disease is treated by drug or procedure",
        head_labels=["Disease"],
        tail_labels=["Drug", "DrugClass", "Procedure"],
        properties=["treatmentLine", "evidenceLevel", "indicationNote"],
    ),
    RelationType(
        type="DIAGNOSED_BY",
        glirel_label="disease is diagnosed by test or procedure",
        head_labels=["Disease"],
        tail_labels=["DiagnosticTest", "Procedure"],
        properties=["diagnosticRole", "goldStandard"],
    ),
    RelationType(
        type="CAUSED_BY",
        glirel_label="disease is caused by pathogen, gene, or mechanism",
        head_labels=["Disease"],
        tail_labels=["Pathogen", "Gene", "MechanismOfAction", "Drug", "RiskFactor"],
    ),
    RelationType(
        type="HAS_GENETIC_CAUSE",
        glirel_label="disease has genetic cause or is associated with gene",
        head_labels=["Disease"],
        tail_labels=["Gene"],
    ),
    RelationType(
        type="AFFECTS",
        glirel_label="disease affects anatomical structure",
        head_labels=["Disease"],
        tail_labels=["AnatomicalStructure"],
    ),
    RelationType(
        type="HAS_COMPLICATION",
        glirel_label="disease has complication or leads to another disease",
        head_labels=["Disease"],
        tail_labels=["Disease"],
        properties=["frequency", "timeframe"],
    ),
    RelationType(
        type="DIFFERENTIAL_FOR",
        glirel_label="disease is differential diagnosis for another disease",
        head_labels=["Disease"],
        tail_labels=["Disease"],
        properties=["distinguishingFeature"],
    ),
    RelationType(
        type="HAS_RISK_FACTOR",
        glirel_label="disease has risk factor",
        head_labels=["Disease"],
        tail_labels=["RiskFactor"],
        properties=["evidenceLevel"],
    ),
    RelationType(
        type="IS_A",
        glirel_label="entity is a subtype or subclass of another",
        head_labels=["Disease", "Drug", "Pathogen"],
        tail_labels=["Disease", "Drug", "Pathogen"],
    ),
    RelationType(
        type="BELONGS_TO_CLASS",
        glirel_label="drug belongs to pharmacological class",
        head_labels=["Drug"],
        tail_labels=["DrugClass"],
    ),
    RelationType(
        type="TARGETS",
        glirel_label="drug targets molecular protein or receptor",
        head_labels=["Drug"],
        tail_labels=["Protein"],
        properties=["interactionType"],
    ),
    RelationType(
        type="INHIBITS",
        glirel_label="drug inhibits or blocks protein or mechanism",
        head_labels=["Drug", "DrugClass"],
        tail_labels=["Protein", "MechanismOfAction"],
    ),
    RelationType(
        type="ACTIVATES",
        glirel_label="drug activates or stimulates protein or mechanism",
        head_labels=["Drug", "DrugClass"],
        tail_labels=["Protein", "MechanismOfAction"],
    ),
    RelationType(
        type="METABOLIZED_BY",
        glirel_label="drug is metabolized by enzyme or protein",
        head_labels=["Drug"],
        tail_labels=["Protein"],
        properties=["metabolismType"],
    ),
    RelationType(
        type="INTERACTS_WITH",
        glirel_label="drug interacts with another drug",
        head_labels=["Drug"],
        tail_labels=["Drug"],
        properties=["interactionType", "severity", "effect"],
    ),
    RelationType(
        type="CONTRAINDICATED_IN",
        glirel_label="drug is contraindicated in disease or condition",
        head_labels=["Drug"],
        tail_labels=["Disease", "RiskFactor"],
        properties=["reason"],
    ),
    RelationType(
        type="CAUSES_ADVERSE_EFFECT",
        glirel_label="drug causes adverse effect or side effect",
        head_labels=["Drug"],
        tail_labels=["Symptom"],
        properties=["frequency", "severity"],
    ),
    RelationType(
        type="MONITORED_BY",
        glirel_label="drug is monitored by diagnostic test",
        head_labels=["Drug"],
        tail_labels=["DiagnosticTest"],
        properties=["monitoringFrequency"],
    ),
    RelationType(
        type="ENCODES",
        glirel_label="gene encodes protein",
        head_labels=["Gene"],
        tail_labels=["Protein"],
    ),
    RelationType(
        type="PARTICIPATES_IN",
        glirel_label="protein participates in biochemical pathway",
        head_labels=["Protein"],
        tail_labels=["Pathway"],
    ),
    RelationType(
        type="PART_OF",
        glirel_label="anatomical structure is part of another structure",
        head_labels=["AnatomicalStructure"],
        tail_labels=["AnatomicalStructure"],
    ),
    RelationType(
        type="INNERVATED_BY",
        glirel_label="anatomical structure is innervated by nerve",
        head_labels=["AnatomicalStructure"],
        tail_labels=["AnatomicalStructure"],
    ),
    RelationType(
        type="SUPPLIED_BY",
        glirel_label="anatomical structure is supplied by vessel",
        head_labels=["AnatomicalStructure"],
        tail_labels=["AnatomicalStructure"],
    ),
    RelationType(
        type="HAS_MECHANISM",
        glirel_label="drug or disease has mechanism of action or pathomechanism",
        head_labels=["Drug", "Disease"],
        tail_labels=["MechanismOfAction"],
    ),
]

MEDICAL_GRAPHRAG_SCHEMA = GraphSchema(
    entity_types=ENTITY_TYPES,
    relation_types=RELATION_TYPES,
)

__all__ = ["ENTITY_TYPES", "RELATION_TYPES", "MEDICAL_GRAPHRAG_SCHEMA"]
