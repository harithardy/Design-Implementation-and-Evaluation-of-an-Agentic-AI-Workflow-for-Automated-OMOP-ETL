from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

try:
    from src.load_tables import DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier
except ModuleNotFoundError:
    from load_tables import DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier

try:
    from src.validation_agent import DEFAULT_REVIEW_TABLE, build_review_table_statement
except ModuleNotFoundError:
    from validation_agent import DEFAULT_REVIEW_TABLE, build_review_table_statement


DEFAULT_TARGET_DATABASE = DEFAULT_DATABASE
DEFAULT_TARGET_SCHEMA = DEFAULT_SCHEMA
REVIEW_SOURCE_PREFIX = "synthea:"

SOURCE_TABLE_NAMES = (
    "patients",
    "organizations",
    "providers",
    "encounters",
    "conditions",
    "procedures",
    "medications",
    "observations",
    "immunizations",
    "allergies",
    "careplans",
    "devices",
    "imaging_studies",
    "supplies",
    "claims",
    "claims_transactions",
    "payers",
    "payer_transitions",
)

SOURCE_TABLE_PREFERENCE = (
    ("synthea", "{name}"),
    ("dbo", "synthea_{name}"),
    ("dbo", "{name}"),
)

RELOAD_TARGET_TABLES = (
    "location",
    "person",
    "death",
    "care_site",
    "provider",
    "visit_occurrence",
    "observation_period",
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
    "measurement",
    "observation",
    "device_exposure",
    "payer_plan_period",
    "cost",
)

LOOKUP_TARGET_TABLES = ("concept", "concept_relationship")

DELETE_ORDER = (
    "cost",
    "observation",
    "measurement",
    "device_exposure",
    "drug_exposure",
    "procedure_occurrence",
    "condition_occurrence",
    "death",
    "observation_period",
    "payer_plan_period",
    "visit_occurrence",
    "person",
    "provider",
    "care_site",
    "location",
)

REQUIRED_VOCABULARIES = ("SNOMED", "RxNorm", "LOINC", "CVX", "UCUM", "Payer")

NO_MATCH_CONCEPT_ID = 0
UNKNOWN_GENDER_CONCEPT_ID = 8551
GENDER_MALE_CONCEPT_ID = 8507
GENDER_FEMALE_CONCEPT_ID = 8532
RACE_WHITE_CONCEPT_ID = 8527
RACE_BLACK_CONCEPT_ID = 8516
RACE_ASIAN_CONCEPT_ID = 8515
RACE_NATIVE_CONCEPT_ID = 8657
RACE_HAWAIIAN_CONCEPT_ID = 8557
RACE_OTHER_CONCEPT_ID = 8522
RACE_UNKNOWN_CONCEPT_ID = 8552
ETHNICITY_HISPANIC_CONCEPT_ID = 38003563
ETHNICITY_NON_HISPANIC_CONCEPT_ID = 38003564
DEATH_TYPE_CONCEPT_ID = 32510
VISIT_TYPE_CONCEPT_ID = 44818518
CONDITION_TYPE_EHR_CONCEPT_ID = 32020
CONDITION_TYPE_CLAIM_CONCEPT_ID = 32810
PROCEDURE_TYPE_CONCEPT_ID = 32827
PROCEDURE_TYPE_IMAGING_CONCEPT_ID = 32841
DRUG_TYPE_MEDICATION_CONCEPT_ID = 32825
DRUG_TYPE_IMMUNIZATION_CONCEPT_ID = 32818
MEASUREMENT_TYPE_CONCEPT_ID = 44818702
OBSERVATION_TYPE_CONCEPT_ID = 38000280
DEVICE_TYPE_CONCEPT_ID = 32818
OBSERVATION_PERIOD_TYPE_CONCEPT_ID = 32817
COST_TYPE_CONCEPT_ID = 32814
COST_TYPE_CLAIM_CONCEPT_ID = 32810


@dataclass(frozen=True)
class SourceTableCheck:
    base_name: str
    schema_name: str | None
    table_name: str | None
    row_count: int
    status: Literal["available", "missing"]
    reason: str


@dataclass(frozen=True)
class TargetTableCheck:
    table_name: str
    role: Literal["reload", "lookup"]
    exists: bool
    row_count: int
    reason: str


@dataclass(frozen=True)
class VocabularyCheck:
    vocabulary_id: str
    concept_count: int
    status: Literal["loaded", "empty"]


@dataclass(frozen=True)
class EtlLoadResult:
    table_name: str
    status: Literal["loaded"]
    rows_loaded: int
    reason: str


class SyntheaMappingState(TypedDict, total=False):
    target_database: str
    target_schema: str
    review_table: str
    execute: bool
    engine: Any
    source_tables: list[SourceTableCheck]
    target_tables: list[TargetTableCheck]
    vocabulary_checks: list[VocabularyCheck]
    execution_results: list[EtlLoadResult]
    review_row_count: int
    summary: dict[str, Any]


def get_engine() -> Any:
    try:
        from src.connection import engine
    except ModuleNotFoundError:
        from connection import engine

    return engine


def build_table_name(schema_name: str, table_name: str) -> str:
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def sql_unicode_literal(value: str) -> str:
    return f"N'{sql_string_literal(value)}'"


def normalize_text_sql(column_sql: str) -> str:
    return f"LTRIM(RTRIM(CONVERT(varchar(255), {column_sql})))"


def gender_concept_sql(column_sql: str) -> str:
    normalized = f"UPPER({normalize_text_sql(column_sql)})"
    return (
        f"CASE {normalized} "
        f"WHEN 'M' THEN {GENDER_MALE_CONCEPT_ID} "
        f"WHEN 'MALE' THEN {GENDER_MALE_CONCEPT_ID} "
        f"WHEN 'F' THEN {GENDER_FEMALE_CONCEPT_ID} "
        f"WHEN 'FEMALE' THEN {GENDER_FEMALE_CONCEPT_ID} "
        f"ELSE {UNKNOWN_GENDER_CONCEPT_ID} END"
    )


def race_concept_sql(column_sql: str) -> str:
    normalized = f"LOWER({normalize_text_sql(column_sql)})"
    return (
        f"CASE {normalized} "
        f"WHEN 'white' THEN {RACE_WHITE_CONCEPT_ID} "
        f"WHEN 'black' THEN {RACE_BLACK_CONCEPT_ID} "
        f"WHEN 'asian' THEN {RACE_ASIAN_CONCEPT_ID} "
        f"WHEN 'native' THEN {RACE_NATIVE_CONCEPT_ID} "
        f"WHEN 'hawaiian' THEN {RACE_HAWAIIAN_CONCEPT_ID} "
        f"WHEN 'other' THEN {RACE_OTHER_CONCEPT_ID} "
        f"ELSE {RACE_UNKNOWN_CONCEPT_ID} END"
    )


def ethnicity_concept_sql(column_sql: str) -> str:
    normalized = f"LOWER({normalize_text_sql(column_sql)})"
    return (
        f"CASE {normalized} "
        f"WHEN 'hispanic' THEN {ETHNICITY_HISPANIC_CONCEPT_ID} "
        f"WHEN 'nonhispanic' THEN {ETHNICITY_NON_HISPANIC_CONCEPT_ID} "
        f"ELSE {NO_MATCH_CONCEPT_ID} END"
    )


def visit_concept_sql(column_sql: str) -> str:
    normalized = f"LOWER({normalize_text_sql(column_sql)})"
    return (
        f"CASE {normalized} "
        f"WHEN 'inpatient' THEN 9201 "
        f"WHEN 'emergency' THEN 9203 "
        f"WHEN 'outpatient' THEN 9202 "
        f"WHEN 'ambulatory' THEN 9202 "
        f"WHEN 'urgentcare' THEN 9202 "
        f"WHEN 'wellness' THEN 9202 "
        f"ELSE {NO_MATCH_CONCEPT_ID} END"
    )


def sql_review_case(condition: str, reason: str) -> str:
    return f"CASE WHEN {condition} THEN {sql_unicode_literal(reason)} ELSE NULL END"


def combine_review_reasons(*cases: str) -> str:
    filtered = [case for case in cases if case]
    if not filtered:
        return "NULL"
    return f"COALESCE({', '.join(filtered)})"


def event_date_review_reason(
    event_date_sql: str,
    birth_date_sql: str,
    field_name: str,
) -> str:
    return combine_review_reasons(
        sql_review_case(f"{event_date_sql} IS NULL", f"{field_name} must not be null."),
        sql_review_case(
            f"{birth_date_sql} IS NOT NULL AND {event_date_sql} < {birth_date_sql}",
            f"{field_name} occurs before birth date.",
        ),
        sql_review_case(
            f"{event_date_sql} > CAST(GETDATE() AS date)",
            f"{field_name} is in the future.",
        ),
    )


def concept_domain_review_reason(
    concept_id_sql: str,
    standard_domain_sql: str,
    expected_domain: str,
    code_label: str,
) -> str:
    return combine_review_reasons(
        sql_review_case(
            f"{concept_id_sql} = {NO_MATCH_CONCEPT_ID}",
            f"{code_label} could not be mapped to a standard {expected_domain} concept.",
        ),
        sql_review_case(
            f"NULLIF({standard_domain_sql}, '') IS NOT NULL AND {standard_domain_sql} <> {sql_unicode_literal(expected_domain)}",
            f"{code_label} mapped to the wrong OMOP domain for {expected_domain}.",
        ),
    )


def build_review_setup_statement(schema_name: str, review_table: str) -> str:
    qualified_name = build_table_name(schema_name, review_table)
    return "\n".join(
        [
            build_review_table_statement(schema_name, review_table),
            f"DELETE FROM {qualified_name} WHERE [source_row_id] LIKE {sql_unicode_literal(REVIEW_SOURCE_PREFIX + '%')};",
        ]
    )


def build_review_insert_from_stage_statement(
    schema_name: str,
    review_table: str,
    target_table: str,
    stage_table: str,
) -> str:
    qualified_name = build_table_name(schema_name, review_table)
    payload_prefix = sql_string_literal(f'{{"stage_table":"{stage_table}","source_row_id":"')
    payload_suffix = sql_string_literal('"}')
    return f"""
WITH review_stage AS (
    SELECT
        *,
        LEFT(
            CONVERT(
                varchar(128),
                COALESCE(
                    source_row_id,
                    CONCAT('synthea:review:{target_table}:', ROW_NUMBER() OVER (ORDER BY (SELECT NULL)))
                )
            ),
            128
        ) AS resolved_source_row_id
    FROM {stage_table}
    WHERE review_reason IS NOT NULL
)
INSERT INTO {qualified_name} (
    [target_table],
    [source_row_id],
    [review_status],
    [review_reason],
    [payload_json]
)
SELECT
    {sql_unicode_literal(target_table)},
    resolved_source_row_id,
    {sql_unicode_literal('review')},
    CONVERT(nvarchar(max), review_reason),
    CONCAT(
        N'{payload_prefix}',
        STRING_ESCAPE(CONVERT(nvarchar(max), resolved_source_row_id), 'json'),
        N'{payload_suffix}'
    )
FROM review_stage;
"""


def preferred_source_table(
    base_name: str,
    available_tables: set[tuple[str, str]],
) -> tuple[str, str] | None:
    for schema_name, pattern in SOURCE_TABLE_PREFERENCE:
        candidate_name = pattern.format(name=base_name)
        if (schema_name, candidate_name) in available_tables:
            return schema_name, candidate_name
    return None


def source_reference_map(source_tables: list[SourceTableCheck]) -> dict[str, str]:
    references: dict[str, str] = {}
    for source_table in source_tables:
        if source_table.status != "available" or source_table.schema_name is None or source_table.table_name is None:
            continue
        references[source_table.base_name] = build_table_name(source_table.schema_name, source_table.table_name)
    return references


def count_rows(connection: Any, schema_name: str, table_name: str) -> int:
    result = connection.execute(text(f"SELECT COUNT(*) FROM {build_table_name(schema_name, table_name)}"))
    return int(result.scalar() or 0)


def metadata_row_count(connection: Any, schema_name: str, table_name: str) -> int:
    result = connection.execute(
        text(
            """
            SELECT COALESCE(SUM(partitions.rows), 0) AS row_count
            FROM sys.partitions AS partitions
            JOIN sys.tables AS tables
              ON tables.object_id = partitions.object_id
            JOIN sys.schemas AS schemas
              ON schemas.schema_id = tables.schema_id
            WHERE schemas.name = :schema_name
              AND tables.name = :table_name
              AND partitions.index_id IN (0, 1)
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    )
    return int(result.scalar() or 0)


def discover_source_tables(state: SyntheaMappingState) -> SyntheaMappingState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_TARGET_DATABASE)
    expected_plain = ", ".join(f"'{sql_string_literal(name)}'" for name in SOURCE_TABLE_NAMES)
    expected_prefixed = ", ".join(
        f"'synthea_{sql_string_literal(name)}'" for name in SOURCE_TABLE_NAMES
    )
    source_tables: list[SourceTableCheck] = []

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))
        rows = connection.execute(
            text(
                f"""
                SELECT s.name AS schema_name, t.name AS table_name
                FROM sys.tables AS t
                JOIN sys.schemas AS s
                  ON s.schema_id = t.schema_id
                WHERE (s.name = 'synthea' AND t.name IN ({expected_plain}))
                   OR (s.name = 'dbo' AND t.name IN ({expected_plain}, {expected_prefixed}))
                """
            )
        )
        available_tables = {(row.schema_name, row.table_name) for row in rows}

        for base_name in SOURCE_TABLE_NAMES:
            resolved = preferred_source_table(base_name, available_tables)
            if resolved is None:
                source_tables.append(
                    SourceTableCheck(
                        base_name=base_name,
                        schema_name=None,
                        table_name=None,
                        row_count=0,
                        status="missing",
                        reason="source table not found in synthea, dbo.synthea_*, or dbo",
                    )
                )
                continue

            schema_name, table_name = resolved
            row_count = metadata_row_count(connection, schema_name, table_name)
            source_tables.append(
                SourceTableCheck(
                    base_name=base_name,
                    schema_name=schema_name,
                    table_name=table_name,
                    row_count=row_count,
                    status="available",
                    reason="resolved using source-table preference order",
                )
            )

    return {"source_tables": source_tables}


def inspect_target_tables(state: SyntheaMappingState) -> SyntheaMappingState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_TARGET_DATABASE)
    target_schema = state.get("target_schema", DEFAULT_TARGET_SCHEMA)
    target_tables: list[TargetTableCheck] = []

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))

        for table_name in RELOAD_TARGET_TABLES:
            exists = bool(
                connection.execute(
                    text(
                        """
                        SELECT 1
                        FROM sys.tables AS t
                        JOIN sys.schemas AS s
                          ON s.schema_id = t.schema_id
                        WHERE s.name = :schema_name
                          AND t.name = :table_name
                        """
                    ),
                    {"schema_name": target_schema, "table_name": table_name},
                ).scalar()
            )
            target_tables.append(
                TargetTableCheck(
                    table_name=table_name,
                    role="reload",
                    exists=exists,
                    row_count=metadata_row_count(connection, target_schema, table_name) if exists else 0,
                    reason="reload target table" if exists else "reload target table missing",
                )
            )

        for table_name in LOOKUP_TARGET_TABLES:
            exists = bool(
                connection.execute(
                    text(
                        """
                        SELECT 1
                        FROM sys.tables AS t
                        JOIN sys.schemas AS s
                          ON s.schema_id = t.schema_id
                        WHERE s.name = :schema_name
                          AND t.name = :table_name
                        """
                    ),
                    {"schema_name": target_schema, "table_name": table_name},
                ).scalar()
            )
            target_tables.append(
                TargetTableCheck(
                    table_name=table_name,
                    role="lookup",
                    exists=exists,
                    row_count=metadata_row_count(connection, target_schema, table_name) if exists else 0,
                    reason="lookup table" if exists else "lookup table missing",
                )
            )

        vocabulary_rows = connection.execute(
            text(
                """
                SELECT vocabulary_id, COUNT(*) AS concept_count
                FROM [dbo].[concept]
                WHERE vocabulary_id IN ('SNOMED', 'RxNorm', 'LOINC', 'CVX', 'UCUM', 'Payer')
                GROUP BY vocabulary_id
                """
            )
        )
        vocabulary_counts = {row.vocabulary_id: int(row.concept_count) for row in vocabulary_rows}

    vocabulary_checks = [
        VocabularyCheck(
            vocabulary_id=vocabulary_id,
            concept_count=vocabulary_counts.get(vocabulary_id, 0),
            status="loaded" if vocabulary_counts.get(vocabulary_id, 0) > 0 else "empty",
        )
        for vocabulary_id in REQUIRED_VOCABULARIES
    ]
    return {"target_tables": target_tables, "vocabulary_checks": vocabulary_checks}


def validate_prerequisites(state: SyntheaMappingState) -> SyntheaMappingState:
    missing_sources = [item.base_name for item in state.get("source_tables", []) if item.status != "available"]
    missing_targets = [item.table_name for item in state.get("target_tables", []) if not item.exists]
    if missing_sources:
        raise ValueError(f"Missing required source tables: {', '.join(missing_sources)}")
    if missing_targets:
        raise ValueError(f"Missing required OMOP tables: {', '.join(missing_targets)}")
    return {}


def should_execute_plan(state: SyntheaMappingState) -> str:
    return "execute" if state.get("execute", True) else "summarize"


def build_delete_statement(schema_name: str, table_name: str) -> str:
    return f"DELETE FROM {build_table_name(schema_name, table_name)}"


def build_concept_map_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #concept_map;
WITH concept_keys AS (
    SELECT DISTINCT vocabulary_id, code
    FROM (
        SELECT 'SNOMED' AS vocabulary_id, LTRIM(RTRIM(CONVERT(varchar(50), CODE))) AS code FROM {source_refs['conditions']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['procedures']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['devices']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['supplies']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['careplans']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), PROCEDURE_CODE))) FROM {source_refs['imaging_studies']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['allergies']}
        UNION ALL
        SELECT 'SNOMED', LTRIM(RTRIM(CONVERT(varchar(50), dx.code)))
        FROM {source_refs['claims']} AS claims
        CROSS APPLY (VALUES
            (claims.DIAGNOSIS1),
            (claims.DIAGNOSIS2),
            (claims.DIAGNOSIS3),
            (claims.DIAGNOSIS4),
            (claims.DIAGNOSIS5),
            (claims.DIAGNOSIS6),
            (claims.DIAGNOSIS7),
            (claims.DIAGNOSIS8)
        ) AS dx(code)
        UNION ALL
        SELECT 'RxNorm', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['medications']}
        UNION ALL
        SELECT 'LOINC', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['observations']}
        UNION ALL
        SELECT 'UCUM', LTRIM(RTRIM(CONVERT(varchar(50), UNITS))) FROM {source_refs['observations']}
        UNION ALL
        SELECT 'CVX', LTRIM(RTRIM(CONVERT(varchar(50), CODE))) FROM {source_refs['immunizations']}
    ) AS raw_codes
    WHERE code IS NOT NULL
      AND code <> ''
)
SELECT
    ck.vocabulary_id,
    ck.code,
    COALESCE(source_match.concept_id, {NO_MATCH_CONCEPT_ID}) AS source_concept_id,
    COALESCE(
        CASE WHEN source_match.standard_concept = 'S' THEN source_match.concept_id END,
        mapped_match.concept_id,
        {NO_MATCH_CONCEPT_ID}
    ) AS standard_concept_id,
    COALESCE(
        CASE WHEN source_match.standard_concept = 'S' THEN source_match.domain_id END,
        mapped_match.domain_id,
        ''
    ) AS standard_domain_id
INTO #concept_map
FROM concept_keys AS ck
OUTER APPLY (
    SELECT TOP (1)
        c.concept_id,
        c.standard_concept,
        c.domain_id
    FROM [dbo].[concept] AS c
    WHERE c.vocabulary_id = ck.vocabulary_id
      AND c.concept_code = ck.code
      AND c.invalid_reason IS NULL
    ORDER BY CASE WHEN c.standard_concept = 'S' THEN 0 ELSE 1 END, c.concept_id
) AS source_match
OUTER APPLY (
    SELECT TOP (1)
        target.concept_id,
        target.domain_id,
        target.standard_concept
    FROM [dbo].[concept_relationship] AS cr
    JOIN [dbo].[concept] AS target
      ON target.concept_id = cr.concept_id_2
     AND target.invalid_reason IS NULL
    WHERE cr.concept_id_1 = source_match.concept_id
      AND cr.relationship_id = 'Maps to'
      AND cr.invalid_reason IS NULL
    ORDER BY CASE WHEN target.standard_concept = 'S' THEN 0 ELSE 1 END, target.concept_id
) AS mapped_match;
CREATE UNIQUE CLUSTERED INDEX IX_concept_map ON #concept_map(vocabulary_id, code);
"""


def build_location_stage_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #location_stage;
WITH location_source AS (
    SELECT
        'patient' AS source_kind,
        patients.Id AS source_id,
        LEFT('PATIENT:' + CONVERT(varchar(36), patients.Id), 50) AS location_source_value,
        LEFT(NULLIF({normalize_text_sql('patients.ADDRESS')}, ''), 50) AS address_1,
        LEFT(NULLIF({normalize_text_sql('patients.CITY')}, ''), 50) AS city,
        LEFT(NULLIF({normalize_text_sql('patients.STATE')}, ''), 2) AS state,
        LEFT(NULLIF({normalize_text_sql('patients.ZIP')}, ''), 9) AS zip,
        LEFT(NULLIF({normalize_text_sql('patients.COUNTY')}, ''), 20) AS county,
        'US' AS country_source_value,
        TRY_CONVERT(float, patients.LAT) AS latitude,
        TRY_CONVERT(float, patients.LON) AS longitude
    FROM {source_refs['patients']} AS patients
    UNION ALL
    SELECT
        'organization',
        organizations.Id,
        LEFT('ORGANIZATION:' + CONVERT(varchar(36), organizations.Id), 50),
        LEFT(NULLIF({normalize_text_sql('organizations.ADDRESS')}, ''), 50),
        LEFT(NULLIF({normalize_text_sql('organizations.CITY')}, ''), 50),
        LEFT(NULLIF({normalize_text_sql('organizations.STATE')}, ''), 2),
        LEFT(NULLIF({normalize_text_sql('organizations.ZIP')}, ''), 9),
        NULL,
        'US',
        TRY_CONVERT(float, organizations.LAT),
        TRY_CONVERT(float, organizations.LON)
    FROM {source_refs['organizations']} AS organizations
)
SELECT
    ROW_NUMBER() OVER (ORDER BY source_kind, source_id) AS location_id,
    source_kind,
    source_id,
    location_source_value,
    address_1,
    city,
    state,
    zip,
    county,
    country_source_value,
    latitude,
    longitude
INTO #location_stage
FROM location_source;
CREATE UNIQUE CLUSTERED INDEX IX_location_stage ON #location_stage(source_kind, source_id);
"""


def build_person_map_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #person_map;
SELECT
    ROW_NUMBER() OVER (ORDER BY patients.Id) AS person_id,
    patients.Id AS patient_source_id,
    locations.location_id,
    patients.BIRTHDATE AS birth_date
INTO #person_map
FROM {source_refs['patients']} AS patients
LEFT JOIN #location_stage AS locations
  ON locations.source_kind = 'patient'
 AND locations.source_id = patients.Id;
CREATE UNIQUE CLUSTERED INDEX IX_person_map ON #person_map(patient_source_id);
"""


def build_insert_location_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'location')} (
    location_id,
    address_1,
    city,
    state,
    zip,
    county,
    location_source_value,
    country_source_value,
    latitude,
    longitude
)
SELECT
    location_id,
    address_1,
    city,
    state,
    zip,
    county,
    location_source_value,
    country_source_value,
    latitude,
    longitude
FROM #location_stage;
"""


def build_insert_person_statement(source_refs: dict[str, str], target_schema: str) -> str:
    gender_sql = gender_concept_sql("patients.GENDER")
    race_sql = race_concept_sql("patients.RACE")
    ethnicity_sql = ethnicity_concept_sql("patients.ETHNICITY")
    return f"""
INSERT INTO {build_table_name(target_schema, 'person')} (
    person_id,
    gender_concept_id,
    year_of_birth,
    month_of_birth,
    day_of_birth,
    birth_datetime,
    race_concept_id,
    ethnicity_concept_id,
    location_id,
    person_source_value,
    gender_source_value,
    gender_source_concept_id,
    race_source_value,
    race_source_concept_id,
    ethnicity_source_value,
    ethnicity_source_concept_id
)
SELECT
    person_map.person_id,
    {gender_sql},
    ISNULL(DATEPART(year, patients.BIRTHDATE), 1900),
    CASE WHEN patients.BIRTHDATE IS NULL THEN NULL ELSE DATEPART(month, patients.BIRTHDATE) END,
    CASE WHEN patients.BIRTHDATE IS NULL THEN NULL ELSE DATEPART(day, patients.BIRTHDATE) END,
    CAST(patients.BIRTHDATE AS datetime),
    {race_sql},
    {ethnicity_sql},
    person_map.location_id,
    LEFT(CONVERT(varchar(50), patients.Id), 50),
    LEFT(NULLIF({normalize_text_sql('patients.GENDER')}, ''), 50),
    {gender_sql},
    LEFT(NULLIF({normalize_text_sql('patients.RACE')}, ''), 50),
    {race_sql},
    LEFT(NULLIF({normalize_text_sql('patients.ETHNICITY')}, ''), 50),
    {ethnicity_sql}
FROM {source_refs['patients']} AS patients
JOIN #person_map AS person_map
  ON person_map.patient_source_id = patients.Id;
"""


def build_death_stage_statement(source_refs: dict[str, str]) -> str:
    death_review_reason = event_date_review_reason(
        "patients.DEATHDATE",
        "person_map.birth_date",
        "death_date",
    )
    return f"""
DROP TABLE IF EXISTS #death_stage;
SELECT
    LEFT('synthea:patients:' + CONVERT(varchar(36), patients.Id) + CHAR(58) + 'death', 128) AS source_row_id,
    person_map.person_id,
    patients.DEATHDATE AS death_date,
    CAST(patients.DEATHDATE AS datetime) AS death_datetime,
    {death_review_reason} AS review_reason
INTO #death_stage
FROM {source_refs['patients']} AS patients
JOIN #person_map AS person_map
  ON person_map.patient_source_id = patients.Id
WHERE patients.DEATHDATE IS NOT NULL;
CREATE UNIQUE CLUSTERED INDEX IX_death_stage ON #death_stage(source_row_id);
"""


def build_insert_death_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'death')} (
    person_id,
    death_date,
    death_datetime,
    death_type_concept_id,
    cause_concept_id
)
SELECT
    person_id,
    death_date,
    death_datetime,
    {DEATH_TYPE_CONCEPT_ID},
    {NO_MATCH_CONCEPT_ID}
FROM #death_stage
WHERE review_reason IS NULL;
"""


def build_care_site_map_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #care_site_map;
SELECT
    ROW_NUMBER() OVER (ORDER BY organizations.Id) AS care_site_id,
    organizations.Id AS organization_source_id,
    locations.location_id
INTO #care_site_map
FROM {source_refs['organizations']} AS organizations
LEFT JOIN #location_stage AS locations
  ON locations.source_kind = 'organization'
 AND locations.source_id = organizations.Id;
CREATE UNIQUE CLUSTERED INDEX IX_care_site_map ON #care_site_map(organization_source_id);
"""


def build_insert_care_site_statement(source_refs: dict[str, str], target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'care_site')} (
    care_site_id,
    care_site_name,
    location_id,
    care_site_source_value
)
SELECT
    care_site_map.care_site_id,
    LEFT(NULLIF({normalize_text_sql('organizations.NAME')}, ''), 255),
    care_site_map.location_id,
    LEFT(CONVERT(varchar(50), organizations.Id), 50)
FROM {source_refs['organizations']} AS organizations
JOIN #care_site_map AS care_site_map
  ON care_site_map.organization_source_id = organizations.Id;
"""


def build_provider_map_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #provider_map;
SELECT
    ROW_NUMBER() OVER (ORDER BY providers.Id) AS provider_id,
    providers.Id AS provider_source_id,
    care_sites.care_site_id
INTO #provider_map
FROM {source_refs['providers']} AS providers
LEFT JOIN #care_site_map AS care_sites
  ON care_sites.organization_source_id = providers.ORGANIZATION;
CREATE UNIQUE CLUSTERED INDEX IX_provider_map ON #provider_map(provider_source_id);
"""


def build_insert_provider_statement(source_refs: dict[str, str], target_schema: str) -> str:
    gender_sql = gender_concept_sql("providers.GENDER")
    return f"""
INSERT INTO {build_table_name(target_schema, 'provider')} (
    provider_id,
    provider_name,
    care_site_id,
    gender_concept_id,
    provider_source_value,
    specialty_source_value,
    gender_source_value,
    gender_source_concept_id
)
SELECT
    provider_map.provider_id,
    LEFT(NULLIF({normalize_text_sql('providers.NAME')}, ''), 255),
    provider_map.care_site_id,
    {gender_sql},
    LEFT(CONVERT(varchar(50), providers.Id), 50),
    LEFT(NULLIF({normalize_text_sql('providers.SPECIALITY')}, ''), 50),
    LEFT(NULLIF({normalize_text_sql('providers.GENDER')}, ''), 50),
    {gender_sql}
FROM {source_refs['providers']} AS providers
JOIN #provider_map AS provider_map
  ON provider_map.provider_source_id = providers.Id;
"""


def build_visit_map_statement(source_refs: dict[str, str]) -> str:
    visit_sql = visit_concept_sql("encounters.ENCOUNTERCLASS")
    visit_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(COALESCE(encounters.START, encounters.STOP) AS date)",
            "person_map.birth_date",
            "visit_start_date",
        ),
        sql_review_case(
            f"{visit_sql} = {NO_MATCH_CONCEPT_ID}",
            "Encounter class could not be mapped to a standard Visit concept.",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #visit_stage;
DROP TABLE IF EXISTS #visit_map;
SELECT
    LEFT('synthea:encounters:' + CONVERT(varchar(36), encounters.Id), 128) AS source_row_id,
    encounters.Id AS encounter_source_id,
    person_map.person_id,
    person_map.birth_date,
    provider_map.provider_id,
    care_site_map.care_site_id,
    CAST(COALESCE(encounters.START, encounters.STOP) AS date) AS visit_start_date,
    CAST(COALESCE(encounters.START, encounters.STOP) AS datetime) AS visit_start_datetime,
    CAST(COALESCE(encounters.STOP, encounters.START) AS date) AS visit_end_date,
    CAST(COALESCE(encounters.STOP, encounters.START) AS datetime) AS visit_end_datetime,
    {visit_sql} AS visit_concept_id,
    LEFT(NULLIF({normalize_text_sql('encounters.ENCOUNTERCLASS')}, ''), 50) AS visit_source_value,
    {visit_review_reason} AS review_reason
INTO #visit_stage
FROM {source_refs['encounters']} AS encounters
JOIN #person_map AS person_map
  ON person_map.patient_source_id = encounters.PATIENT
LEFT JOIN #provider_map AS provider_map
  ON provider_map.provider_source_id = encounters.PROVIDER
LEFT JOIN #care_site_map AS care_site_map
  ON care_site_map.organization_source_id = encounters.ORGANIZATION;
CREATE UNIQUE CLUSTERED INDEX IX_visit_stage ON #visit_stage(encounter_source_id);

WITH valid_rows AS (
    SELECT *
    FROM #visit_stage
    WHERE review_reason IS NULL
),
numbered AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY person_id, visit_start_datetime, encounter_source_id
        ) AS visit_occurrence_id,
        *
    FROM valid_rows
),
sequenced AS (
    SELECT
        *,
        LAG(visit_occurrence_id) OVER (
            PARTITION BY person_id
            ORDER BY visit_start_datetime, encounter_source_id
        ) AS preceding_visit_occurrence_id
    FROM numbered
)
SELECT
    source_row_id,
    encounter_source_id,
    person_id,
    birth_date,
    provider_id,
    care_site_id,
    visit_start_date,
    visit_start_datetime,
    visit_end_date,
    visit_end_datetime,
    visit_concept_id,
    visit_source_value,
    review_reason,
    visit_occurrence_id,
    preceding_visit_occurrence_id
INTO #visit_map
FROM sequenced;
CREATE UNIQUE CLUSTERED INDEX IX_visit_map ON #visit_map(encounter_source_id);
"""


def build_insert_visit_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'visit_occurrence')} (
    visit_occurrence_id,
    person_id,
    visit_concept_id,
    visit_start_date,
    visit_start_datetime,
    visit_end_date,
    visit_end_datetime,
    visit_type_concept_id,
    provider_id,
    care_site_id,
    visit_source_value,
    preceding_visit_occurrence_id
)
SELECT
    visit_occurrence_id,
    person_id,
    visit_concept_id,
    visit_start_date,
    visit_start_datetime,
    visit_end_date,
    visit_end_datetime,
    {VISIT_TYPE_CONCEPT_ID},
    provider_id,
    care_site_id,
    visit_source_value,
    preceding_visit_occurrence_id
FROM #visit_map;
"""


def build_insert_observation_period_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'observation_period')} (
    observation_period_id,
    person_id,
    observation_period_start_date,
    observation_period_end_date,
    period_type_concept_id
)
SELECT
    ROW_NUMBER() OVER (ORDER BY person_id) AS observation_period_id,
    person_id,
    MIN(visit_start_date) AS observation_period_start_date,
    MAX(visit_end_date) AS observation_period_end_date,
    {OBSERVATION_PERIOD_TYPE_CONCEPT_ID} AS period_type_concept_id
FROM #visit_map
GROUP BY person_id;
"""


def build_condition_stage_statement(source_refs: dict[str, str]) -> str:
    condition_review_reason = combine_review_reasons(
        event_date_review_reason(
            "COALESCE(conditions.START, visit_map.visit_start_date)",
            "person_map.birth_date",
            "condition_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Condition",
            "Condition code",
        ),
    )
    claim_condition_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(claims.SERVICEDATE AS date)",
            "person_map.birth_date",
            "condition_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Condition",
            "Claim diagnosis code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #condition_stage;
WITH condition_source AS (
    SELECT
        'conditions' AS source_table,
        LEFT(
            'synthea:conditions:' + CONVERT(varchar(36), conditions.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), conditions.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(10), conditions.START, 23) + CHAR(58) +
            LEFT(CONVERT(varchar(50), conditions.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        COALESCE(conditions.START, visit_map.visit_start_date) AS condition_start_date,
        COALESCE(CAST(conditions.START AS datetime), visit_map.visit_start_datetime) AS condition_start_datetime,
        conditions.STOP AS condition_end_date,
        CAST(conditions.STOP AS datetime) AS condition_end_datetime,
        COALESCE(visit_map.provider_id, NULL) AS provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), conditions.CODE), 50) AS condition_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS condition_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS condition_concept_id,
        {CONDITION_TYPE_EHR_CONCEPT_ID} AS condition_type_concept_id,
        {condition_review_reason} AS review_reason
    FROM {source_refs['conditions']} AS conditions
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = conditions.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = conditions.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), conditions.CODE)))
    WHERE COALESCE(conditions.START, visit_map.visit_start_date) IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), conditions.CODE))), '') IS NOT NULL
    UNION ALL
    SELECT
        'claims' AS source_table,
        LEFT('synthea:claims:' + CONVERT(varchar(36), claims.Id) + CHAR(58) + 'dx' + diagnosis.slot_number, 128) AS source_row_id,
        person_map.person_id,
        CAST(claims.SERVICEDATE AS date) AS condition_start_date,
        CAST(claims.SERVICEDATE AS datetime) AS condition_start_datetime,
        NULL AS condition_end_date,
        NULL AS condition_end_datetime,
        COALESCE(visit_map.provider_id, provider_map.provider_id) AS provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), diagnosis.code), 50) AS condition_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS condition_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS condition_concept_id,
        {CONDITION_TYPE_CLAIM_CONCEPT_ID} AS condition_type_concept_id,
        {claim_condition_review_reason} AS review_reason
    FROM {source_refs['claims']} AS claims
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = claims.PATIENTID
    LEFT JOIN #provider_map AS provider_map
      ON provider_map.provider_source_id = claims.PROVIDERID
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = claims.APPOINTMENTID
    CROSS APPLY (VALUES
        (claims.DIAGNOSIS1, '1'),
        (claims.DIAGNOSIS2, '2'),
        (claims.DIAGNOSIS3, '3'),
        (claims.DIAGNOSIS4, '4'),
        (claims.DIAGNOSIS5, '5'),
        (claims.DIAGNOSIS6, '6'),
        (claims.DIAGNOSIS7, '7'),
        (claims.DIAGNOSIS8, '8')
    ) AS diagnosis(code, slot_number)
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), diagnosis.code)))
    WHERE claims.SERVICEDATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), diagnosis.code))), '') IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, condition_start_datetime, source_table, condition_source_value
    ) AS condition_occurrence_id,
    *
INTO #condition_stage
FROM condition_source;
CREATE UNIQUE CLUSTERED INDEX IX_condition_stage ON #condition_stage(condition_occurrence_id);
CREATE NONCLUSTERED INDEX IX_condition_stage_source_row_id ON #condition_stage(source_row_id);
"""


def build_insert_condition_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'condition_occurrence')} (
    condition_occurrence_id,
    person_id,
    condition_concept_id,
    condition_start_date,
    condition_start_datetime,
    condition_end_date,
    condition_end_datetime,
    condition_type_concept_id,
    provider_id,
    visit_occurrence_id,
    condition_source_value,
    condition_source_concept_id
)
SELECT
    condition_occurrence_id,
    person_id,
    condition_concept_id,
    condition_start_date,
    condition_start_datetime,
    condition_end_date,
    condition_end_datetime,
    condition_type_concept_id,
    provider_id,
    visit_occurrence_id,
    condition_source_value,
    condition_source_concept_id
FROM #condition_stage
WHERE review_reason IS NULL;
"""


def build_procedure_stage_statement(source_refs: dict[str, str]) -> str:
    procedure_review_reason = combine_review_reasons(
        event_date_review_reason(
            "COALESCE(CAST(procedures.START AS date), visit_map.visit_start_date)",
            "person_map.birth_date",
            "procedure_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Procedure",
            "Procedure code",
        ),
    )
    careplan_review_reason = combine_review_reasons(
        event_date_review_reason(
            "COALESCE(careplans.START, visit_map.visit_start_date)",
            "person_map.birth_date",
            "procedure_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Procedure",
            "Care plan code",
        ),
    )
    imaging_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(imaging.DATE AS date)",
            "person_map.birth_date",
            "procedure_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Procedure",
            "Imaging procedure code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #procedure_stage;
WITH procedure_source AS (
    SELECT
        'procedures' AS source_table,
        LEFT(
            'synthea:procedures:' + CONVERT(varchar(36), procedures.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), procedures.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(30), procedures.START, 126) + CHAR(58) +
            LEFT(CONVERT(varchar(50), procedures.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        COALESCE(CAST(procedures.START AS date), visit_map.visit_start_date) AS procedure_date,
        COALESCE(CAST(procedures.START AS datetime), visit_map.visit_start_datetime) AS procedure_datetime,
        CAST(procedures.STOP AS date) AS procedure_end_date,
        CAST(procedures.STOP AS datetime) AS procedure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), procedures.CODE), 50) AS procedure_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_concept_id,
        {PROCEDURE_TYPE_CONCEPT_ID} AS procedure_type_concept_id,
        NULL AS modifier_source_value,
        TRY_CONVERT(float, procedures.BASE_COST) AS raw_total_cost,
        {procedure_review_reason} AS review_reason
    FROM {source_refs['procedures']} AS procedures
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = procedures.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = procedures.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), procedures.CODE)))
    WHERE COALESCE(procedures.START, visit_map.visit_start_date) IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), procedures.CODE))), '') IS NOT NULL
    UNION ALL
    SELECT
        'careplans' AS source_table,
        LEFT('synthea:careplans:' + CONVERT(varchar(36), careplans.Id), 128) AS source_row_id,
        person_map.person_id,
        COALESCE(careplans.START, visit_map.visit_start_date) AS procedure_date,
        CAST(COALESCE(careplans.START, visit_map.visit_start_date) AS datetime) AS procedure_datetime,
        careplans.STOP AS procedure_end_date,
        CAST(careplans.STOP AS datetime) AS procedure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), careplans.CODE), 50) AS procedure_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_concept_id,
        {PROCEDURE_TYPE_CONCEPT_ID} AS procedure_type_concept_id,
        NULL AS modifier_source_value,
        NULL AS raw_total_cost,
        {careplan_review_reason} AS review_reason
    FROM {source_refs['careplans']} AS careplans
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = careplans.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = careplans.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), careplans.CODE)))
    WHERE COALESCE(careplans.START, visit_map.visit_start_date) IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), careplans.CODE))), '') IS NOT NULL
    UNION ALL
    SELECT
        'imaging_studies' AS source_table,
        LEFT('synthea:imaging:' + CONVERT(varchar(36), imaging.Id), 128) AS source_row_id,
        person_map.person_id,
        CAST(imaging.DATE AS date) AS procedure_date,
        CAST(imaging.DATE AS datetime) AS procedure_datetime,
        NULL AS procedure_end_date,
        NULL AS procedure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), imaging.PROCEDURE_CODE), 50) AS procedure_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS procedure_concept_id,
        {PROCEDURE_TYPE_IMAGING_CONCEPT_ID} AS procedure_type_concept_id,
        LEFT(NULLIF({normalize_text_sql('imaging.MODALITY_CODE')}, ''), 50) AS modifier_source_value,
        NULL AS raw_total_cost,
        {imaging_review_reason} AS review_reason
    FROM {source_refs['imaging_studies']} AS imaging
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = imaging.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = imaging.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), imaging.PROCEDURE_CODE)))
    WHERE imaging.DATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), imaging.PROCEDURE_CODE))), '') IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, procedure_datetime, source_table, procedure_source_value
    ) AS procedure_occurrence_id,
    *
INTO #procedure_stage
FROM procedure_source;
CREATE UNIQUE CLUSTERED INDEX IX_procedure_stage ON #procedure_stage(procedure_occurrence_id);
CREATE NONCLUSTERED INDEX IX_procedure_stage_source_row_id ON #procedure_stage(source_row_id);
"""


def build_insert_procedure_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'procedure_occurrence')} (
    procedure_occurrence_id,
    person_id,
    procedure_concept_id,
    procedure_date,
    procedure_datetime,
    procedure_end_date,
    procedure_end_datetime,
    procedure_type_concept_id,
    provider_id,
    visit_occurrence_id,
    procedure_source_value,
    procedure_source_concept_id,
    modifier_source_value
)
SELECT
    procedure_occurrence_id,
    person_id,
    procedure_concept_id,
    procedure_date,
    procedure_datetime,
    procedure_end_date,
    procedure_end_datetime,
    procedure_type_concept_id,
    provider_id,
    visit_occurrence_id,
    procedure_source_value,
    procedure_source_concept_id,
    modifier_source_value
FROM #procedure_stage
WHERE review_reason IS NULL;
"""


def build_drug_stage_statement(source_refs: dict[str, str]) -> str:
    medication_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(COALESCE(medications.START, medications.STOP) AS date)",
            "person_map.birth_date",
            "drug_exposure_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Drug",
            "Medication code",
        ),
    )
    immunization_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(immunizations.DATE AS date)",
            "person_map.birth_date",
            "drug_exposure_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Drug",
            "Immunization code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #drug_stage;
WITH drug_source AS (
    SELECT
        'medications' AS source_table,
        LEFT(
            'synthea:medications:' + CONVERT(varchar(36), medications.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), medications.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(30), medications.START, 126) + CHAR(58) +
            LEFT(CONVERT(varchar(50), medications.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(COALESCE(medications.START, medications.STOP) AS date) AS drug_exposure_start_date,
        CAST(COALESCE(medications.START, medications.STOP) AS datetime) AS drug_exposure_start_datetime,
        CAST(COALESCE(medications.STOP, medications.START) AS date) AS drug_exposure_end_date,
        CAST(COALESCE(medications.STOP, medications.START) AS datetime) AS drug_exposure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), medications.CODE), 50) AS drug_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS drug_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS drug_concept_id,
        {DRUG_TYPE_MEDICATION_CONCEPT_ID} AS drug_type_concept_id,
        CASE
            WHEN TRY_CONVERT(int, medications.DISPENSES) IS NULL THEN NULL
            WHEN TRY_CONVERT(int, medications.DISPENSES) <= 1 THEN 0
            ELSE TRY_CONVERT(int, medications.DISPENSES) - 1
        END AS refills,
        TRY_CONVERT(float, medications.DISPENSES) AS quantity,
        CASE
            WHEN medications.START IS NULL AND medications.STOP IS NULL THEN NULL
            WHEN DATEDIFF(day, COALESCE(medications.START, medications.STOP), COALESCE(medications.STOP, medications.START)) <= 0 THEN 1
            ELSE DATEDIFF(day, COALESCE(medications.START, medications.STOP), COALESCE(medications.STOP, medications.START))
        END AS days_supply,
        TRY_CONVERT(float, medications.TOTALCOST) AS raw_total_cost,
        TRY_CONVERT(float, medications.PAYER_COVERAGE) AS raw_paid_by_payer,
        {medication_review_reason} AS review_reason
    FROM {source_refs['medications']} AS medications
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = medications.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = medications.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'RxNorm'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), medications.CODE)))
    WHERE COALESCE(medications.START, medications.STOP) IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), medications.CODE))), '') IS NOT NULL
    UNION ALL
    SELECT
        'immunizations' AS source_table,
        LEFT(
            'synthea:immunizations:' + CONVERT(varchar(36), immunizations.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), immunizations.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(30), immunizations.DATE, 126) + CHAR(58) +
            LEFT(CONVERT(varchar(50), immunizations.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(immunizations.DATE AS date) AS drug_exposure_start_date,
        CAST(immunizations.DATE AS datetime) AS drug_exposure_start_datetime,
        CAST(immunizations.DATE AS date) AS drug_exposure_end_date,
        CAST(immunizations.DATE AS datetime) AS drug_exposure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), immunizations.CODE), 50) AS drug_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS drug_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS drug_concept_id,
        {DRUG_TYPE_IMMUNIZATION_CONCEPT_ID} AS drug_type_concept_id,
        0 AS refills,
        1.0 AS quantity,
        1 AS days_supply,
        TRY_CONVERT(float, immunizations.BASE_COST) AS raw_total_cost,
        NULL AS raw_paid_by_payer,
        {immunization_review_reason} AS review_reason
    FROM {source_refs['immunizations']} AS immunizations
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = immunizations.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = immunizations.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'CVX'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), immunizations.CODE)))
    WHERE immunizations.DATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), immunizations.CODE))), '') IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, drug_exposure_start_datetime, source_table, drug_source_value
    ) AS drug_exposure_id,
    *
INTO #drug_stage
FROM drug_source;
CREATE UNIQUE CLUSTERED INDEX IX_drug_stage ON #drug_stage(drug_exposure_id);
CREATE NONCLUSTERED INDEX IX_drug_stage_source_row_id ON #drug_stage(source_row_id);
"""


def build_insert_drug_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'drug_exposure')} (
    drug_exposure_id,
    person_id,
    drug_concept_id,
    drug_exposure_start_date,
    drug_exposure_start_datetime,
    drug_exposure_end_date,
    drug_exposure_end_datetime,
    drug_type_concept_id,
    refills,
    quantity,
    days_supply,
    provider_id,
    visit_occurrence_id,
    drug_source_value,
    drug_source_concept_id
)
SELECT
    drug_exposure_id,
    person_id,
    drug_concept_id,
    drug_exposure_start_date,
    drug_exposure_start_datetime,
    drug_exposure_end_date,
    drug_exposure_end_datetime,
    drug_type_concept_id,
    refills,
    quantity,
    days_supply,
    provider_id,
    visit_occurrence_id,
    drug_source_value,
    drug_source_concept_id
FROM #drug_stage
WHERE review_reason IS NULL;
"""


def build_measurement_stage_statement(source_refs: dict[str, str]) -> str:
    measurement_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(observations.DATE AS date)",
            "person_map.birth_date",
            "measurement_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "code_map.standard_domain_id",
            "Measurement",
            "Observation code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #measurement_stage;
WITH measurement_source AS (
    SELECT
        LEFT(
            'synthea:observations:' + CONVERT(varchar(36), observations.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), observations.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(30), observations.DATE, 126) + CHAR(58) +
            LEFT(CONVERT(varchar(50), observations.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(observations.DATE AS date) AS measurement_date,
        CAST(observations.DATE AS datetime) AS measurement_datetime,
        CONVERT(varchar(8), CAST(observations.DATE AS time), 108) AS measurement_time,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), observations.CODE), 50) AS measurement_source_value,
        COALESCE(code_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS measurement_source_concept_id,
        COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS measurement_concept_id,
        TRY_CONVERT(float, observations.VALUE) AS value_as_number,
        COALESCE(unit_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS unit_concept_id,
        COALESCE(unit_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS unit_source_concept_id,
        LEFT(NULLIF({normalize_text_sql('observations.UNITS')}, ''), 50) AS unit_source_value,
        LEFT(NULLIF({normalize_text_sql('observations.VALUE')}, ''), 50) AS value_source_value,
        {measurement_review_reason} AS review_reason
    FROM {source_refs['observations']} AS observations
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = observations.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = observations.ENCOUNTER
    LEFT JOIN #concept_map AS code_map
      ON code_map.vocabulary_id = 'LOINC'
     AND code_map.code = LTRIM(RTRIM(CONVERT(varchar(50), observations.CODE)))
    LEFT JOIN #concept_map AS unit_map
      ON unit_map.vocabulary_id = 'UCUM'
     AND unit_map.code = LTRIM(RTRIM(CONVERT(varchar(50), observations.UNITS)))
    WHERE observations.DATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), observations.CODE))), '') IS NOT NULL
      AND (
          NULLIF(code_map.standard_domain_id, '') = 'Measurement'
          OR (
              NULLIF(code_map.standard_domain_id, '') IS NULL
              AND UPPER(NULLIF({normalize_text_sql('observations.TYPE')}, '')) = 'NUMERIC'
          )
      )
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, measurement_datetime, measurement_source_value
    ) AS measurement_id,
    *
INTO #measurement_stage
FROM measurement_source;
CREATE UNIQUE CLUSTERED INDEX IX_measurement_stage ON #measurement_stage(measurement_id);
CREATE NONCLUSTERED INDEX IX_measurement_stage_source_row_id ON #measurement_stage(source_row_id);
"""


def build_insert_measurement_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'measurement')} (
    measurement_id,
    person_id,
    measurement_concept_id,
    measurement_date,
    measurement_datetime,
    measurement_time,
    measurement_type_concept_id,
    value_as_number,
    unit_concept_id,
    provider_id,
    visit_occurrence_id,
    measurement_source_value,
    measurement_source_concept_id,
    unit_source_value,
    unit_source_concept_id,
    value_source_value
)
SELECT
    measurement_id,
    person_id,
    measurement_concept_id,
    measurement_date,
    measurement_datetime,
    measurement_time,
    {MEASUREMENT_TYPE_CONCEPT_ID},
    value_as_number,
    NULLIF(unit_concept_id, {NO_MATCH_CONCEPT_ID}),
    provider_id,
    visit_occurrence_id,
    measurement_source_value,
    measurement_source_concept_id,
    unit_source_value,
    NULLIF(unit_source_concept_id, {NO_MATCH_CONCEPT_ID}),
    value_source_value
FROM #measurement_stage
WHERE review_reason IS NULL;
"""


def build_observation_stage_statement(source_refs: dict[str, str]) -> str:
    observation_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(observations.DATE AS date)",
            "person_map.birth_date",
            "observation_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "code_map.standard_domain_id",
            "Observation",
            "Observation code",
        ),
    )
    allergy_review_reason = combine_review_reasons(
        event_date_review_reason(
            "allergies.START",
            "person_map.birth_date",
            "observation_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "code_map.standard_domain_id",
            "Observation",
            "Allergy code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #observation_stage;
WITH observation_source AS (
    SELECT
        'observations' AS source_table,
        LEFT(
            'synthea:observations:' + CONVERT(varchar(36), observations.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), observations.ENCOUNTER) + CHAR(58) +
            CONVERT(varchar(30), observations.DATE, 126) + CHAR(58) +
            LEFT(CONVERT(varchar(50), observations.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(observations.DATE AS date) AS observation_date,
        CAST(observations.DATE AS datetime) AS observation_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), observations.CODE), 50) AS observation_source_value,
        COALESCE(code_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS observation_source_concept_id,
        COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS observation_concept_id,
        TRY_CONVERT(float, observations.VALUE) AS value_as_number,
        LEFT(NULLIF(CONVERT(varchar(60), observations.VALUE), ''), 60) AS value_as_string,
        COALESCE(unit_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS unit_concept_id,
        LEFT(NULLIF({normalize_text_sql('observations.UNITS')}, ''), 50) AS unit_source_value,
        LEFT(NULLIF({normalize_text_sql('observations.VALUE')}, ''), 50) AS value_source_value,
        LEFT(NULLIF({normalize_text_sql('observations.CATEGORY')}, ''), 50) AS qualifier_source_value,
        {observation_review_reason} AS review_reason
    FROM {source_refs['observations']} AS observations
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = observations.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = observations.ENCOUNTER
    LEFT JOIN #concept_map AS code_map
      ON code_map.vocabulary_id = 'LOINC'
     AND code_map.code = LTRIM(RTRIM(CONVERT(varchar(50), observations.CODE)))
    LEFT JOIN #concept_map AS unit_map
      ON unit_map.vocabulary_id = 'UCUM'
     AND unit_map.code = LTRIM(RTRIM(CONVERT(varchar(50), observations.UNITS)))
    WHERE observations.DATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), observations.CODE))), '') IS NOT NULL
      AND NOT (
          NULLIF(code_map.standard_domain_id, '') = 'Measurement'
          OR (
              NULLIF(code_map.standard_domain_id, '') IS NULL
              AND UPPER(NULLIF({normalize_text_sql('observations.TYPE')}, '')) = 'NUMERIC'
          )
      )
    UNION ALL
    SELECT
        'allergies' AS source_table,
        LEFT('synthea:allergies:' + CONVERT(varchar(36), allergies.PATIENT) + CHAR(58) + CONVERT(varchar(36), allergies.ENCOUNTER) + CHAR(58) + LEFT(CONVERT(varchar(50), allergies.CODE), 20), 128) AS source_row_id,
        person_map.person_id,
        allergies.START AS observation_date,
        CAST(allergies.START AS datetime) AS observation_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), allergies.CODE), 50) AS observation_source_value,
        COALESCE(code_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS observation_source_concept_id,
        COALESCE(code_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS observation_concept_id,
        NULL AS value_as_number,
        LEFT(NULLIF(CONVERT(varchar(60), allergies.DESCRIPTION), ''), 60) AS value_as_string,
        NULL AS unit_concept_id,
        NULL AS unit_source_value,
        LEFT(NULLIF({normalize_text_sql('allergies.DESCRIPTION')}, ''), 50) AS value_source_value,
        LEFT(
            COALESCE(
                NULLIF({normalize_text_sql('allergies.TYPE')}, ''),
                NULLIF({normalize_text_sql('allergies.CATEGORY')}, '')
            ),
            50
        ) AS qualifier_source_value,
        {allergy_review_reason} AS review_reason
    FROM {source_refs['allergies']} AS allergies
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = allergies.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = allergies.ENCOUNTER
    LEFT JOIN #concept_map AS code_map
      ON code_map.vocabulary_id = 'SNOMED'
     AND code_map.code = LTRIM(RTRIM(CONVERT(varchar(50), allergies.CODE)))
    WHERE allergies.START IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), allergies.CODE))), '') IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, observation_datetime, source_table, observation_source_value
    ) AS observation_id,
    *
INTO #observation_stage
FROM observation_source;
CREATE UNIQUE CLUSTERED INDEX IX_observation_stage ON #observation_stage(observation_id);
CREATE NONCLUSTERED INDEX IX_observation_stage_source_row_id ON #observation_stage(source_row_id);
"""


def build_insert_observation_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'observation')} (
    observation_id,
    person_id,
    observation_concept_id,
    observation_date,
    observation_datetime,
    observation_type_concept_id,
    value_as_number,
    value_as_string,
    unit_concept_id,
    provider_id,
    visit_occurrence_id,
    observation_source_value,
    observation_source_concept_id,
    unit_source_value,
    qualifier_source_value,
    value_source_value
)
SELECT
    observation_id,
    person_id,
    observation_concept_id,
    observation_date,
    observation_datetime,
    {OBSERVATION_TYPE_CONCEPT_ID},
    value_as_number,
    value_as_string,
    NULLIF(unit_concept_id, {NO_MATCH_CONCEPT_ID}),
    provider_id,
    visit_occurrence_id,
    observation_source_value,
    observation_source_concept_id,
    unit_source_value,
    qualifier_source_value,
    value_source_value
FROM #observation_stage
WHERE review_reason IS NULL;
"""


def build_device_stage_statement(source_refs: dict[str, str]) -> str:
    device_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(COALESCE(devices.START, devices.STOP) AS date)",
            "person_map.birth_date",
            "device_exposure_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Device",
            "Device code",
        ),
    )
    supply_review_reason = combine_review_reasons(
        event_date_review_reason(
            "supplies.DATE",
            "person_map.birth_date",
            "device_exposure_start_date",
        ),
        concept_domain_review_reason(
            f"COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID})",
            "concept_map.standard_domain_id",
            "Device",
            "Supply code",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #device_stage;
WITH device_source AS (
    SELECT
        'devices' AS source_table,
        LEFT(
            'synthea:devices:' + CONVERT(varchar(36), devices.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), devices.ENCOUNTER) + CHAR(58) +
            LEFT(CONVERT(varchar(50), devices.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(COALESCE(devices.START, devices.STOP) AS date) AS device_exposure_start_date,
        CAST(COALESCE(devices.START, devices.STOP) AS datetime) AS device_exposure_start_datetime,
        CAST(devices.STOP AS date) AS device_exposure_end_date,
        CAST(devices.STOP AS datetime) AS device_exposure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), devices.CODE), 50) AS device_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS device_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS device_concept_id,
        LEFT(NULLIF(CONVERT(varchar(255), devices.UDI), ''), 255) AS unique_device_id,
        1 AS quantity,
        {device_review_reason} AS review_reason
    FROM {source_refs['devices']} AS devices
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = devices.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = devices.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), devices.CODE)))
    WHERE COALESCE(devices.START, devices.STOP) IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), devices.CODE))), '') IS NOT NULL
    UNION ALL
    SELECT
        'supplies' AS source_table,
        LEFT(
            'synthea:supplies:' + CONVERT(varchar(36), supplies.PATIENT) + CHAR(58) +
            CONVERT(varchar(36), supplies.ENCOUNTER) + CHAR(58) +
            LEFT(CONVERT(varchar(50), supplies.CODE), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        supplies.DATE AS device_exposure_start_date,
        CAST(supplies.DATE AS datetime) AS device_exposure_start_datetime,
        NULL AS device_exposure_end_date,
        NULL AS device_exposure_end_datetime,
        visit_map.provider_id,
        visit_map.visit_occurrence_id,
        LEFT(CONVERT(varchar(50), supplies.CODE), 50) AS device_source_value,
        COALESCE(concept_map.source_concept_id, {NO_MATCH_CONCEPT_ID}) AS device_source_concept_id,
        COALESCE(concept_map.standard_concept_id, {NO_MATCH_CONCEPT_ID}) AS device_concept_id,
        NULL AS unique_device_id,
        TRY_CONVERT(int, supplies.QUANTITY) AS quantity,
        {supply_review_reason} AS review_reason
    FROM {source_refs['supplies']} AS supplies
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = supplies.PATIENT
    LEFT JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = supplies.ENCOUNTER
    LEFT JOIN #concept_map AS concept_map
      ON concept_map.vocabulary_id = 'SNOMED'
     AND concept_map.code = LTRIM(RTRIM(CONVERT(varchar(50), supplies.CODE)))
    WHERE supplies.DATE IS NOT NULL
      AND NULLIF(LTRIM(RTRIM(CONVERT(varchar(50), supplies.CODE))), '') IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, device_exposure_start_datetime, source_table, device_source_value
    ) AS device_exposure_id,
    *
INTO #device_stage
FROM device_source;
CREATE UNIQUE CLUSTERED INDEX IX_device_stage ON #device_stage(device_exposure_id);
CREATE NONCLUSTERED INDEX IX_device_stage_source_row_id ON #device_stage(source_row_id);
"""


def build_insert_device_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'device_exposure')} (
    device_exposure_id,
    person_id,
    device_concept_id,
    device_exposure_start_date,
    device_exposure_start_datetime,
    device_exposure_end_date,
    device_exposure_end_datetime,
    device_type_concept_id,
    unique_device_id,
    quantity,
    provider_id,
    visit_occurrence_id,
    device_source_value,
    device_source_concept_id
)
SELECT
    device_exposure_id,
    person_id,
    device_concept_id,
    device_exposure_start_date,
    device_exposure_start_datetime,
    device_exposure_end_date,
    device_exposure_end_datetime,
    {DEVICE_TYPE_CONCEPT_ID},
    unique_device_id,
    quantity,
    provider_id,
    visit_occurrence_id,
    device_source_value,
    device_source_concept_id
FROM #device_stage
WHERE review_reason IS NULL;
"""


def build_payer_stage_statement(source_refs: dict[str, str]) -> str:
    payer_review_reason = combine_review_reasons(
        event_date_review_reason(
            "CAST(payer_transitions.START_YEAR AS date)",
            "person_map.birth_date",
            "payer_plan_period_start_date",
        ),
        sql_review_case(
            f"COALESCE(payer_map.payer_concept_id, {NO_MATCH_CONCEPT_ID}) = {NO_MATCH_CONCEPT_ID}",
            "Payer could not be mapped to a standard Payer concept.",
        ),
        sql_review_case(
            "CAST(COALESCE(payer_transitions.END_YEAR, payer_transitions.START_YEAR) AS date) < CAST(payer_transitions.START_YEAR AS date)",
            "payer_plan_period_end_date occurs before payer_plan_period_start_date.",
        ),
    )
    return f"""
DROP TABLE IF EXISTS #payer_map;
SELECT
    payers.Id AS payer_source_id,
    LEFT(NULLIF({normalize_text_sql('payers.NAME')}, ''), 50) AS payer_source_value,
    COALESCE(payer_concept.concept_id, {NO_MATCH_CONCEPT_ID}) AS payer_concept_id
INTO #payer_map
FROM {source_refs['payers']} AS payers
OUTER APPLY (
    SELECT TOP (1) concept.concept_id
    FROM [dbo].[concept] AS concept
    WHERE concept.vocabulary_id = 'Payer'
      AND concept.invalid_reason IS NULL
      AND concept.concept_name = CONVERT(varchar(255), payers.NAME)
    ORDER BY concept.concept_id
) AS payer_concept;
CREATE UNIQUE CLUSTERED INDEX IX_payer_map ON #payer_map(payer_source_id);

DROP TABLE IF EXISTS #payer_plan_stage;
WITH payer_source AS (
    SELECT
        LEFT(
            'synthea:payer_transitions:' + CONVERT(varchar(36), payer_transitions.PATIENT) + CHAR(58) +
            CONVERT(varchar(30), payer_transitions.START_YEAR) + CHAR(58) +
            LEFT(CONVERT(varchar(50), payer_transitions.PAYER), 20),
            128
        ) AS source_row_id,
        person_map.person_id,
        CAST(payer_transitions.START_YEAR AS date) AS payer_plan_period_start_date,
        CAST(COALESCE(payer_transitions.END_YEAR, payer_transitions.START_YEAR) AS date) AS payer_plan_period_end_date,
        COALESCE(payer_map.payer_concept_id, {NO_MATCH_CONCEPT_ID}) AS payer_concept_id,
        COALESCE(
            payer_map.payer_source_value,
            LEFT(CONVERT(varchar(50), payer_transitions.PAYER), 50)
        ) AS payer_source_value,
        LEFT(CONVERT(varchar(50), payer_transitions.MEMBERID), 50) AS plan_source_value,
        LEFT(NULLIF({normalize_text_sql('payer_transitions.OWNERSHIP')}, ''), 50) AS sponsor_source_value,
        LEFT(NULLIF({normalize_text_sql('payer_transitions.OWNERNAME')}, ''), 50) AS family_source_value,
        LEFT(NULLIF({normalize_text_sql('payer_transitions.SECONDARY_PAYER')}, ''), 50) AS stop_reason_source_value,
        {payer_review_reason} AS review_reason
    FROM {source_refs['payer_transitions']} AS payer_transitions
    JOIN #person_map AS person_map
      ON person_map.patient_source_id = payer_transitions.PATIENT
    LEFT JOIN #payer_map AS payer_map
      ON payer_map.payer_source_id = payer_transitions.PAYER
    WHERE payer_transitions.START_YEAR IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY person_id, payer_plan_period_start_date, payer_source_value, plan_source_value
    ) AS payer_plan_period_id,
    *
INTO #payer_plan_stage
FROM payer_source;
CREATE UNIQUE CLUSTERED INDEX IX_payer_plan_stage ON #payer_plan_stage(payer_plan_period_id);
CREATE NONCLUSTERED INDEX IX_payer_plan_stage_source_row_id ON #payer_plan_stage(source_row_id);
"""


def build_insert_payer_plan_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'payer_plan_period')} (
    payer_plan_period_id,
    person_id,
    payer_plan_period_start_date,
    payer_plan_period_end_date,
    payer_concept_id,
    payer_source_value,
    plan_source_value,
    sponsor_source_value,
    family_source_value,
    stop_reason_source_value
)
SELECT
    payer_plan_period_id,
    person_id,
    payer_plan_period_start_date,
    payer_plan_period_end_date,
    payer_concept_id,
    payer_source_value,
    plan_source_value,
    sponsor_source_value,
    family_source_value,
    stop_reason_source_value
FROM #payer_plan_stage
WHERE review_reason IS NULL;
"""


def build_cost_stage_statement(source_refs: dict[str, str]) -> str:
    return f"""
DROP TABLE IF EXISTS #cost_stage;
WITH cost_source AS (
    SELECT
        visit_map.visit_occurrence_id AS cost_event_id,
        'Visit' AS cost_domain_id,
        {COST_TYPE_CONCEPT_ID} AS cost_type_concept_id,
        TRY_CONVERT(float, encounters.TOTAL_CLAIM_COST) AS total_charge,
        TRY_CONVERT(float, encounters.BASE_ENCOUNTER_COST) AS total_cost,
        NULL AS total_paid,
        TRY_CONVERT(float, encounters.PAYER_COVERAGE) AS paid_by_payer,
        CASE
            WHEN TRY_CONVERT(float, encounters.TOTAL_CLAIM_COST) IS NULL THEN NULL
            WHEN TRY_CONVERT(float, encounters.PAYER_COVERAGE) IS NULL THEN TRY_CONVERT(float, encounters.TOTAL_CLAIM_COST)
            ELSE TRY_CONVERT(float, encounters.TOTAL_CLAIM_COST) - TRY_CONVERT(float, encounters.PAYER_COVERAGE)
        END AS paid_by_patient
    FROM {source_refs['encounters']} AS encounters
    JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = encounters.Id
    UNION ALL
    SELECT
        procedure_occurrence_id AS cost_event_id,
        'Procedure' AS cost_domain_id,
        {COST_TYPE_CONCEPT_ID} AS cost_type_concept_id,
        raw_total_cost AS total_charge,
        raw_total_cost AS total_cost,
        NULL AS total_paid,
        NULL AS paid_by_payer,
        raw_total_cost AS paid_by_patient
    FROM #procedure_stage
    WHERE source_table = 'procedures'
      AND raw_total_cost IS NOT NULL
      AND review_reason IS NULL
    UNION ALL
    SELECT
        drug_exposure_id AS cost_event_id,
        'Drug' AS cost_domain_id,
        {COST_TYPE_CONCEPT_ID} AS cost_type_concept_id,
        raw_total_cost AS total_charge,
        raw_total_cost AS total_cost,
        NULL AS total_paid,
        raw_paid_by_payer AS paid_by_payer,
        CASE
            WHEN raw_total_cost IS NULL THEN NULL
            WHEN raw_paid_by_payer IS NULL THEN raw_total_cost
            ELSE raw_total_cost - raw_paid_by_payer
        END AS paid_by_patient
    FROM #drug_stage
    WHERE raw_total_cost IS NOT NULL
      AND review_reason IS NULL
    UNION ALL
    SELECT
        visit_map.visit_occurrence_id AS cost_event_id,
        'Visit' AS cost_domain_id,
        {COST_TYPE_CLAIM_CONCEPT_ID} AS cost_type_concept_id,
        TRY_CONVERT(float, claim_transactions.AMOUNT) AS total_charge,
        TRY_CONVERT(float, claim_transactions.AMOUNT) AS total_cost,
        TRY_CONVERT(float, claim_transactions.PAYMENTS) AS total_paid,
        NULL AS paid_by_payer,
        TRY_CONVERT(float, claim_transactions.OUTSTANDING) AS paid_by_patient
    FROM {source_refs['claims_transactions']} AS claim_transactions
    JOIN #visit_map AS visit_map
      ON visit_map.encounter_source_id = claim_transactions.APPOINTMENTID
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY cost_domain_id, cost_event_id
    ) AS cost_id,
    cost_event_id,
    cost_domain_id,
    cost_type_concept_id,
    total_charge,
    total_cost,
    total_paid,
    paid_by_payer,
    paid_by_patient
INTO #cost_stage
FROM cost_source
WHERE cost_event_id IS NOT NULL
  AND (
      total_charge IS NOT NULL
      OR total_cost IS NOT NULL
      OR total_paid IS NOT NULL
      OR paid_by_payer IS NOT NULL
      OR paid_by_patient IS NOT NULL
  );
CREATE UNIQUE CLUSTERED INDEX IX_cost_stage ON #cost_stage(cost_id);
"""


def build_insert_cost_statement(target_schema: str) -> str:
    return f"""
INSERT INTO {build_table_name(target_schema, 'cost')} (
    cost_id,
    cost_event_id,
    cost_domain_id,
    cost_type_concept_id,
    total_charge,
    total_cost,
    total_paid,
    paid_by_payer,
    paid_by_patient
)
SELECT
    cost_id,
    cost_event_id,
    cost_domain_id,
    cost_type_concept_id,
    total_charge,
    total_cost,
    total_paid,
    paid_by_payer,
    paid_by_patient
FROM #cost_stage;
"""


def count_target_tables(connection: Any, target_schema: str) -> list[EtlLoadResult]:
    results: list[EtlLoadResult] = []
    for table_name in RELOAD_TARGET_TABLES:
        results.append(
            EtlLoadResult(
                table_name=table_name,
                status="loaded",
                rows_loaded=count_rows(connection, target_schema, table_name),
                reason="loaded from Synthea source tables",
            )
        )
    return results


def count_review_rows(connection: Any, schema_name: str, review_table: str) -> int:
    result = connection.execute(
        text(
            f"""
            SELECT COUNT(*)
            FROM {build_table_name(schema_name, review_table)}
            WHERE source_row_id LIKE :source_prefix
            """
        ),
        {"source_prefix": f"{REVIEW_SOURCE_PREFIX}%"},
    )
    return int(result.scalar() or 0)


def execute_etl(state: SyntheaMappingState) -> SyntheaMappingState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_TARGET_DATABASE)
    target_schema = state.get("target_schema", DEFAULT_TARGET_SCHEMA)
    review_table = state.get("review_table", DEFAULT_REVIEW_TABLE)
    source_refs = source_reference_map(state["source_tables"])

    statements = [
        build_review_setup_statement(target_schema, review_table),
        build_concept_map_statement(source_refs),
        build_location_stage_statement(source_refs),
        build_person_map_statement(source_refs),
        build_death_stage_statement(source_refs),
        build_care_site_map_statement(source_refs),
        build_provider_map_statement(source_refs),
        build_visit_map_statement(source_refs),
        build_condition_stage_statement(source_refs),
        build_procedure_stage_statement(source_refs),
        build_drug_stage_statement(source_refs),
        build_measurement_stage_statement(source_refs),
        build_observation_stage_statement(source_refs),
        build_device_stage_statement(source_refs),
        build_payer_stage_statement(source_refs),
        build_review_insert_from_stage_statement(target_schema, review_table, "death", "#death_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "visit_occurrence", "#visit_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "condition_occurrence", "#condition_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "procedure_occurrence", "#procedure_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "drug_exposure", "#drug_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "measurement", "#measurement_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "observation", "#observation_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "device_exposure", "#device_stage"),
        build_review_insert_from_stage_statement(target_schema, review_table, "payer_plan_period", "#payer_plan_stage"),
        build_insert_location_statement(target_schema),
        build_insert_person_statement(source_refs, target_schema),
        build_insert_death_statement(target_schema),
        build_insert_care_site_statement(source_refs, target_schema),
        build_insert_provider_statement(source_refs, target_schema),
        build_insert_visit_statement(target_schema),
        build_insert_observation_period_statement(target_schema),
        build_insert_condition_statement(target_schema),
        build_insert_procedure_statement(target_schema),
        build_insert_drug_statement(target_schema),
        build_insert_measurement_statement(target_schema),
        build_insert_observation_statement(target_schema),
        build_insert_device_statement(target_schema),
        build_insert_payer_plan_statement(target_schema),
        build_cost_stage_statement(source_refs),
        build_insert_cost_statement(target_schema),
    ]

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))
        for table_name in DELETE_ORDER:
            connection.execute(text(build_delete_statement(target_schema, table_name)))
        for statement in statements:
            connection.execute(text(statement))
        execution_results = count_target_tables(connection, target_schema)
        review_row_count = count_review_rows(connection, target_schema, review_table)

    return {"execution_results": execution_results, "review_row_count": review_row_count}


def summarize_run(state: SyntheaMappingState) -> SyntheaMappingState:
    execution_results = state.get("execution_results", [])
    rows_by_table = {
        result.table_name: result.rows_loaded
        for result in execution_results
    }
    vocabulary_counts = {
        check.vocabulary_id: check.concept_count
        for check in state.get("vocabulary_checks", [])
    }
    review_row_count = int(state.get("review_row_count", 0))
    return {
        "summary": {
            "target_database": state.get("target_database", DEFAULT_TARGET_DATABASE),
            "target_schema": state.get("target_schema", DEFAULT_TARGET_SCHEMA),
            "review_table": state.get("review_table", DEFAULT_REVIEW_TABLE),
            "source_tables_checked": len(state.get("source_tables", [])),
            "source_rows_checked": sum(item.row_count for item in state.get("source_tables", [])),
            "target_tables_checked": len(state.get("target_tables", [])),
            "tables_loaded": len(execution_results),
            "rows_loaded": sum(result.rows_loaded for result in execution_results),
            "review_rows": review_row_count,
            "rows_by_table": rows_by_table,
            "resolved_sources": {
                item.base_name: (
                    f"{item.schema_name}.{item.table_name}" if item.schema_name and item.table_name else "missing"
                )
                for item in state.get("source_tables", [])
            },
            "vocabulary_counts": vocabulary_counts,
        }
    }


def build_synthea_mapping_graph():
    workflow = StateGraph(SyntheaMappingState)
    workflow.add_node("discover", discover_source_tables)
    workflow.add_node("inspect", inspect_target_tables)
    workflow.add_node("validate", validate_prerequisites)
    workflow.add_node("execute", execute_etl)
    workflow.add_node("summarize", summarize_run)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "inspect")
    workflow.add_edge("inspect", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_execute_plan,
        {"execute": "execute", "summarize": "summarize"},
    )
    workflow.add_edge("execute", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


synthea_mapping_agent = build_synthea_mapping_graph()


def run_synthea_mapping_agent(
    target_database: str = DEFAULT_TARGET_DATABASE,
    target_schema: str = DEFAULT_TARGET_SCHEMA,
    review_table: str = DEFAULT_REVIEW_TABLE,
    execute: bool = True,
    engine: Any | None = None,
) -> SyntheaMappingState:
    initial_state: SyntheaMappingState = {
        "target_database": target_database,
        "target_schema": target_schema,
        "review_table": review_table,
        "execute": execute,
    }
    if engine is not None:
        initial_state["engine"] = engine
    return synthea_mapping_agent.invoke(initial_state)


def render_summary(state: SyntheaMappingState) -> str:
    summary = state.get("summary", {})
    rows_by_table = summary.get("rows_by_table", {})
    vocabulary_counts = summary.get("vocabulary_counts", {})
    return "\n".join(
        [
            f"Database: {summary.get('target_database', DEFAULT_TARGET_DATABASE)}",
            f"Schema: {summary.get('target_schema', DEFAULT_TARGET_SCHEMA)}",
            f"Review table: {summary.get('review_table', DEFAULT_REVIEW_TABLE)}",
            f"Source tables checked: {summary.get('source_tables_checked', 0)}",
            f"Source rows checked: {summary.get('source_rows_checked', 0)}",
            f"Target tables checked: {summary.get('target_tables_checked', 0)}",
            f"Tables loaded: {summary.get('tables_loaded', 0)}",
            f"Rows loaded: {summary.get('rows_loaded', 0)}",
            f"Rows routed to review: {summary.get('review_rows', 0)}",
            (
                "Vocabulary counts: "
                + ", ".join(f"{vocabulary_id}={count}" for vocabulary_id, count in vocabulary_counts.items())
                if vocabulary_counts
                else "Vocabulary counts: none"
            ),
            (
                "Rows by table: "
                + ", ".join(f"{table_name}={count}" for table_name, count in rows_by_table.items())
                if rows_by_table
                else "Rows by table: none"
            ),
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the staged Synthea source tables in AI_OMOP, validate the required OMOP "
            "targets, and load the OMOP clinical tables with a full set-based ETL."
        )
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_TARGET_DATABASE,
        help="Target SQL Server database. Defaults to AI_OMOP.",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_TARGET_SCHEMA,
        help="Target OMOP schema. Defaults to dbo.",
    )
    parser.add_argument(
        "--review-table",
        default=DEFAULT_REVIEW_TABLE,
        help="Review table name for uncertain or invalid mappings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check source and target tables without modifying OMOP data.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_synthea_mapping_agent(
        target_database=args.database,
        target_schema=args.schema,
        review_table=args.review_table,
        execute=not args.dry_run,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
