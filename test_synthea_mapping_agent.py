from types import SimpleNamespace

import pytest

from src.synthea_mapping_agent import (
    LOOKUP_TARGET_TABLES,
    RELOAD_TARGET_TABLES,
    REQUIRED_VOCABULARIES,
    SOURCE_TABLE_NAMES,
    SourceTableCheck,
    TargetTableCheck,
    build_insert_device_statement,
    build_insert_measurement_statement,
    build_insert_observation_statement,
    build_insert_payer_plan_statement,
    build_review_insert_from_stage_statement,
    build_review_setup_statement,
    preferred_source_table,
    render_summary,
    run_synthea_mapping_agent,
    source_reference_map,
    validate_prerequisites,
)


def test_preferred_source_table_uses_expected_resolution_order() -> None:
    available_tables = {
        ("dbo", "patients"),
        ("dbo", "synthea_patients"),
        ("synthea", "patients"),
    }

    assert preferred_source_table("patients", available_tables) == ("synthea", "patients")
    assert preferred_source_table("medications", {("dbo", "synthea_medications")}) == (
        "dbo",
        "synthea_medications",
    )
    assert preferred_source_table("observations", {("dbo", "observations")}) == (
        "dbo",
        "observations",
    )


def test_source_reference_map_uses_only_available_tables() -> None:
    references = source_reference_map(
        [
            SourceTableCheck("patients", "synthea", "patients", 1163, "available", "ok"),
            SourceTableCheck("claims", "dbo", "synthea_claims", 100, "available", "ok"),
            SourceTableCheck("devices", None, None, 0, "missing", "missing"),
        ]
    )

    assert references == {
        "patients": "[synthea].[patients]",
        "claims": "[dbo].[synthea_claims]",
    }


def test_validate_prerequisites_rejects_missing_sources_and_targets() -> None:
    with pytest.raises(ValueError, match="Missing required source tables: devices"):
        validate_prerequisites(
            {
                "source_tables": [
                    SourceTableCheck("patients", "synthea", "patients", 1, "available", "ok"),
                    SourceTableCheck("devices", None, None, 0, "missing", "missing"),
                ],
                "target_tables": [],
            }
        )

    with pytest.raises(ValueError, match="Missing required OMOP tables: person"):
        validate_prerequisites(
            {
                "source_tables": [
                    SourceTableCheck("patients", "synthea", "patients", 1, "available", "ok"),
                ],
                "target_tables": [
                    TargetTableCheck("person", "reload", False, 0, "missing"),
                ],
            }
        )


def test_run_synthea_mapping_agent_dry_run_summarizes_checked_tables() -> None:
    fake_engine = FakeEngine()

    state = run_synthea_mapping_agent(
        target_database="AI_OMOP",
        target_schema="dbo",
        review_table="etl_validation_review",
        execute=False,
        engine=fake_engine,
    )

    summary = state["summary"]
    assert summary["source_tables_checked"] == len(SOURCE_TABLE_NAMES)
    assert summary["target_tables_checked"] == len(RELOAD_TARGET_TABLES) + len(LOOKUP_TARGET_TABLES)
    assert summary["tables_loaded"] == 0
    assert summary["source_rows_checked"] > 0
    assert summary["vocabulary_counts"]["SNOMED"] == 1000
    assert summary["review_table"] == "etl_validation_review"
    assert summary["review_rows"] == 0

    rendered = render_summary(state)
    assert "Database: AI_OMOP" in rendered
    assert "Review table: etl_validation_review" in rendered
    assert "Tables loaded: 0" in rendered
    assert "Rows routed to review: 0" in rendered
    assert "Vocabulary counts:" in rendered


def test_review_setup_statement_clears_only_synthea_rows() -> None:
    statement = build_review_setup_statement("dbo", "etl_validation_review")

    assert "CREATE TABLE [dbo].[etl_validation_review]" in statement
    assert "DELETE FROM [dbo].[etl_validation_review]" in statement
    assert "synthea:%" in statement


def test_review_insert_statement_synthesizes_missing_source_row_ids() -> None:
    statement = build_review_insert_from_stage_statement(
        "dbo",
        "etl_validation_review",
        "measurement",
        "#measurement_stage",
    )

    assert "WITH review_stage AS (" in statement
    assert "resolved_source_row_id" in statement
    assert "synthea:review:measurement:" in statement
    assert "FROM review_stage;" in statement


@pytest.mark.parametrize(
    ("builder", "table_name"),
    [
        (build_insert_measurement_statement, "measurement"),
        (build_insert_observation_statement, "observation"),
        (build_insert_device_statement, "device_exposure"),
        (build_insert_payer_plan_statement, "payer_plan_period"),
    ],
)
def test_review_filtered_insert_statements(builder, table_name: str) -> None:
    statement = builder("dbo")

    assert f"INSERT INTO [dbo].[{table_name}]" in statement
    assert "WHERE review_reason IS NULL" in statement
    assert ";\nWHERE" not in statement


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeIterableResult:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def scalar(self):
        if not self.rows:
            return None
        first = self.rows[0]
        if isinstance(first, tuple):
            return first[0]
        return first


class FakeConnection:
    source_counts = {
        table_name: index + 1 for index, table_name in enumerate(SOURCE_TABLE_NAMES)
    }
    target_counts = {
        table_name: 0 for table_name in (*RELOAD_TARGET_TABLES, *LOOKUP_TARGET_TABLES)
    }
    vocabulary_counts = {
        "SNOMED": 1000,
        "RxNorm": 500,
        "LOINC": 400,
        "CVX": 50,
        "UCUM": 20,
        "Payer": 10,
    }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        sql = str(statement).strip()
        params = params or {}

        if sql.startswith("USE "):
            return FakeScalarResult(None)

        if "WHERE (s.name = 'synthea' AND t.name IN" in sql:
            rows = [
                SimpleNamespace(schema_name="synthea", table_name=table_name)
                for table_name in SOURCE_TABLE_NAMES
            ]
            return FakeIterableResult(rows)

        if "WHERE s.name = :schema_name" in sql and "AND t.name = :table_name" in sql:
            table_name = params["table_name"]
            exists = table_name in self.target_counts
            return FakeScalarResult(1 if exists else None)

        if "FROM sys.partitions AS partitions" in sql:
            table_name = params["table_name"]
            if table_name in self.source_counts:
                return FakeScalarResult(self.source_counts[table_name])
            return FakeScalarResult(self.target_counts.get(table_name, 0))

        if sql.startswith("SELECT COUNT(*) FROM "):
            table_name = sql.split(".")[-1].strip("[]")
            if table_name in self.source_counts:
                return FakeScalarResult(self.source_counts[table_name])
            return FakeScalarResult(self.target_counts.get(table_name, 0))

        if "GROUP BY vocabulary_id" in sql and "FROM [dbo].[concept]" in sql:
            rows = [
                SimpleNamespace(vocabulary_id=vocabulary_id, concept_count=count)
                for vocabulary_id, count in self.vocabulary_counts.items()
            ]
            return FakeIterableResult(rows)

        raise AssertionError(f"Unhandled SQL in fake connection:\n{sql}\nparams={params}")


class FakeEngine:
    def connect(self):
        return FakeConnection()
