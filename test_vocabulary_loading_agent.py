from pathlib import Path
import re

import pytest

from src.vocabulary_loading_agent import (
    TABLE_DELETE_ORDER,
    TABLE_LOAD_ORDER,
    VocabularyBulkLoadPlan,
    build_bulk_insert_statement,
    discover_vocabulary_files,
    execute_vocabulary_loads,
    plan_vocabulary_loads,
)


def write_csv(path: Path, content: str = "header\nvalue\n") -> None:
    path.write_text(content, encoding="utf-8")


def test_plan_vocabulary_loads_uses_exact_fk_order_and_ignores_unrequested_files(
    tmp_path: Path,
) -> None:
    write_csv(tmp_path / "DOMAIN.csv")
    write_csv(tmp_path / "VOCABULARY.csv")
    write_csv(tmp_path / "CONCEPT_CLASS.csv")
    write_csv(tmp_path / "CONCEPT.csv")
    write_csv(tmp_path / "CONCEPT_CPT4.csv")
    write_csv(tmp_path / "DRUG_STRENGTH.csv")

    discovered_state = discover_vocabulary_files({"vocabulary_directory": tmp_path})
    planned_state = plan_vocabulary_loads(discovered_state)
    plans_by_table = {
        plan.table_name: [file.name for file in plan.source_files]
        for plan in planned_state["load_plans"]
    }

    assert [plan.table_name for plan in planned_state["load_plans"]] == list(TABLE_LOAD_ORDER)
    assert plans_by_table["domain"] == ["DOMAIN.csv"]
    assert plans_by_table["vocabulary"] == ["VOCABULARY.csv"]
    assert plans_by_table["concept_class"] == ["CONCEPT_CLASS.csv"]
    assert plans_by_table["concept"] == ["CONCEPT.csv", "CONCEPT_CPT4.csv"]
    assert plans_by_table["drug_strength"] == ["DRUG_STRENGTH.csv"]
    assert plans_by_table["concept_ancestor"] == []
    assert planned_state["ignored_files"] == []


def test_plan_vocabulary_loads_fails_if_any_csv_is_unmapped(tmp_path: Path) -> None:
    write_csv(tmp_path / "DOMAIN.csv")
    write_csv(tmp_path / "MYSTERY.csv")

    discovered_state = discover_vocabulary_files({"vocabulary_directory": tmp_path})

    with pytest.raises(ValueError, match="Unmapped vocabulary CSV files were found: MYSTERY.csv"):
        plan_vocabulary_loads(discovered_state)


def test_build_bulk_insert_statement_uses_requested_sql_server_options(tmp_path: Path) -> None:
    source_file = tmp_path / "VOCABULARY.csv"
    write_csv(source_file)

    statement = build_bulk_insert_statement("dbo", "vocabulary", source_file)

    assert "BULK INSERT [dbo].[vocabulary]" in statement
    assert f"FROM N'{source_file.resolve()}'" in statement
    assert "FIELDTERMINATOR = '\\t'" in statement
    assert "ROWTERMINATOR = '0x0A'" in statement
    assert "FIRSTROW = 2" in statement
    assert "CODEPAGE = '65001'" in statement
    assert "TABLOCK" in statement


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeConnection:
    bulk_insert_re = re.compile(
        r"BULK INSERT \[dbo\]\.\[([^\]]+)\].*?FROM N'([^']+)'",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.table_counts = {table_name: 9 for table_name in TABLE_LOAD_ORDER}
        self.bulk_file_rows = {
            "DOMAIN.csv": 3,
            "VOCABULARY.csv": 2,
            "CONCEPT_CPT4.csv": 5,
            "DRUG_STRENGTH.csv": 4,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement):
        sql_text = str(statement).strip()
        self.executed_sql.append(sql_text)

        if sql_text.startswith("SELECT COUNT(*) FROM "):
            table_name = sql_text.split(".")[-1].strip("[]")
            return FakeResult(self.table_counts[table_name])

        if sql_text.startswith("DELETE FROM "):
            table_name = sql_text.split(".")[-1].strip("[]")
            self.table_counts[table_name] = 0
            return FakeResult(None)

        bulk_match = self.bulk_insert_re.search(sql_text)
        if bulk_match:
            table_name = bulk_match.group(1)
            file_name = Path(bulk_match.group(2)).name
            self.table_counts[table_name] += self.bulk_file_rows[file_name]
            return FakeResult(None)

        return FakeResult(None)


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_execute_vocabulary_loads_deletes_in_reverse_order_bulk_loads_in_order_and_counts_rows(
    tmp_path: Path,
) -> None:
    domain_file = tmp_path / "DOMAIN.csv"
    vocabulary_file = tmp_path / "VOCABULARY.csv"
    concept_file = tmp_path / "CONCEPT_CPT4.csv"
    drug_strength_file = tmp_path / "DRUG_STRENGTH.csv"
    write_csv(domain_file)
    write_csv(vocabulary_file)
    write_csv(concept_file)
    write_csv(drug_strength_file)

    fake_engine = FakeEngine()
    state = execute_vocabulary_loads(
        {
            "engine": fake_engine,
            "target_database": "AI_OMOP",
            "schema": "dbo",
            "load_plans": [
                VocabularyBulkLoadPlan("domain", (domain_file,)),
                VocabularyBulkLoadPlan("vocabulary", (vocabulary_file,)),
                VocabularyBulkLoadPlan("concept_class", ()),
                VocabularyBulkLoadPlan("relationship", ()),
                VocabularyBulkLoadPlan("concept", (concept_file,)),
                VocabularyBulkLoadPlan("concept_synonym", ()),
                VocabularyBulkLoadPlan("concept_relationship", ()),
                VocabularyBulkLoadPlan("drug_strength", (drug_strength_file,)),
                VocabularyBulkLoadPlan("concept_ancestor", ()),
            ],
        }
    )

    results = state["execution_results"]
    assert [result.table_name for result in results] == list(TABLE_LOAD_ORDER)
    assert [result.status for result in results] == [
        "loaded",
        "loaded",
        "skipped",
        "skipped",
        "loaded",
        "skipped",
        "skipped",
        "loaded",
        "skipped",
    ]
    assert results[0].rows_loaded == 3
    assert results[1].rows_loaded == 2
    assert results[4].rows_loaded == 5
    assert results[7].rows_loaded == 4
    assert results[-1].rows_loaded == 0

    executed_sql = fake_engine.connection.executed_sql
    assert executed_sql[0] == "USE [AI_OMOP]"
    assert executed_sql[1:1 + len(TABLE_LOAD_ORDER)] == [
        f"ALTER TABLE [dbo].[{table_name}] CHECK CONSTRAINT ALL"
        for table_name in TABLE_LOAD_ORDER
    ]
    delete_start = 1 + len(TABLE_LOAD_ORDER)
    assert executed_sql[delete_start:delete_start + len(TABLE_DELETE_ORDER)] == [
        f"DELETE FROM [dbo].[{table_name}]"
        for table_name in TABLE_DELETE_ORDER
    ]
    assert any("BULK INSERT [dbo].[domain]" in sql for sql in executed_sql)
    assert any("BULK INSERT [dbo].[vocabulary]" in sql for sql in executed_sql)
    assert any("BULK INSERT [dbo].[concept]" in sql for sql in executed_sql)
    assert any("BULK INSERT [dbo].[drug_strength]" in sql for sql in executed_sql)
