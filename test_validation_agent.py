from datetime import date
from pathlib import Path
from types import SimpleNamespace

import src.validation_agent as validation_module
from src.validation_agent import (
    ConceptBinding,
    ProposedInsertRow,
    render_summary,
    run_validation_agent,
)


def test_validation_agent_approves_valid_rows_and_routes_invalid_rows_to_review() -> None:
    state = run_validation_agent(
        proposed_rows=[
            ProposedInsertRow(
                target_table="condition_occurrence",
                source_row_id="conditions-1",
                values={
                    "person_id": 1,
                    "condition_concept_id": 1001,
                    "condition_source_concept_id": 1002,
                    "condition_start_date": "2020-01-05",
                },
                concept_bindings=(
                    ConceptBinding("condition_concept_id", 1001, requires_standard=True),
                    ConceptBinding("condition_source_concept_id", 1002, requires_standard=False, required_domain="Condition"),
                ),
                start_date_field="condition_start_date",
            ),
            ProposedInsertRow(
                target_table="procedure_occurrence",
                source_row_id="procedures-1",
                values={
                    "person_id": 1,
                    "procedure_concept_id": 2001,
                    "procedure_date": "2020-01-06",
                },
                concept_bindings=(
                    ConceptBinding("procedure_concept_id", 2001, requires_standard=True),
                ),
                start_date_field="procedure_date",
            ),
            ProposedInsertRow(
                target_table="observation",
                source_row_id="observations-1",
                values={
                    "person_id": 1,
                    "observation_concept_id": 3001,
                    "observation_date": "2020-01-07",
                },
                concept_bindings=(
                    ConceptBinding("observation_concept_id", 3001, requires_standard=True),
                ),
                start_date_field="observation_date",
                uncertainty_reason="text value could map to observation or measurement",
            ),
        ],
        execute=False,
        engine=FakeEngine(),
    )

    approved_sql = state["generated_sql"]["approved_sql"]
    review_sql = state["generated_sql"]["review_sql"]
    row_results = {result.source_row_id: result for result in state["row_results"]}

    assert row_results["conditions-1"].status == "approved"
    assert row_results["procedures-1"].status == "review"
    assert "expected Procedure" in row_results["procedures-1"].reason
    assert row_results["observations-1"].status == "review"
    assert "Uncertain mapping" in row_results["observations-1"].reason

    assert "INSERT INTO [dbo].[condition_occurrence]" in approved_sql
    assert "conditions-1" not in review_sql
    assert "INSERT INTO [dbo].[procedure_occurrence]" not in approved_sql
    assert "INSERT INTO [dbo].[etl_validation_review]" in review_sql
    assert "procedures-1" in review_sql
    assert "observations-1" in review_sql


def test_validation_agent_rejects_missing_person_and_bad_event_dates() -> None:
    state = run_validation_agent(
        proposed_rows=[
            ProposedInsertRow(
                target_table="measurement",
                source_row_id="measurements-missing-person",
                values={
                    "person_id": 99,
                    "measurement_concept_id": 4001,
                    "measurement_date": "2020-01-08",
                },
                concept_bindings=(ConceptBinding("measurement_concept_id", 4001, requires_standard=True),),
                start_date_field="measurement_date",
            ),
            ProposedInsertRow(
                target_table="drug_exposure",
                source_row_id="drugs-before-birth",
                values={
                    "person_id": 1,
                    "drug_concept_id": 5001,
                    "drug_exposure_start_date": "1999-12-31",
                },
                concept_bindings=(ConceptBinding("drug_concept_id", 5001, requires_standard=True),),
                start_date_field="drug_exposure_start_date",
            ),
            ProposedInsertRow(
                target_table="observation",
                source_row_id="observation-future",
                values={
                    "person_id": 1,
                    "observation_concept_id": 3001,
                    "observation_date": "2999-01-01",
                },
                concept_bindings=(ConceptBinding("observation_concept_id", 3001, requires_standard=True),),
                start_date_field="observation_date",
            ),
        ],
        execute=False,
        engine=FakeEngine(),
    )

    row_results = {result.source_row_id: result for result in state["row_results"]}

    assert row_results["measurements-missing-person"].status == "review"
    assert "person_id 99 does not exist" in row_results["measurements-missing-person"].reason
    assert row_results["drugs-before-birth"].status == "review"
    assert "occurs before birth date" in row_results["drugs-before-birth"].reason
    assert row_results["observation-future"].status == "review"
    assert "is in the future" in row_results["observation-future"].reason


def test_validation_agent_rejects_nonstandard_and_missing_concepts() -> None:
    state = run_validation_agent(
        proposed_rows=[
            ProposedInsertRow(
                target_table="drug_exposure",
                source_row_id="drug-nonstandard",
                values={
                    "person_id": 1,
                    "drug_concept_id": 5002,
                    "drug_exposure_start_date": "2020-01-10",
                },
                concept_bindings=(ConceptBinding("drug_concept_id", 5002, requires_standard=True),),
                start_date_field="drug_exposure_start_date",
            ),
            ProposedInsertRow(
                target_table="condition_occurrence",
                source_row_id="condition-missing",
                values={
                    "person_id": 1,
                    "condition_concept_id": 9999,
                    "condition_start_date": "2020-01-11",
                },
                concept_bindings=(ConceptBinding("condition_concept_id", 9999, requires_standard=True),),
                start_date_field="condition_start_date",
            ),
        ],
        execute=False,
        engine=FakeEngine(),
    )

    concept_results = {
        (result.source_row_id, result.field_name): result
        for result in state["concept_results"]
    }

    assert concept_results[("drug-nonstandard", "drug_concept_id")].status == "review"
    assert "not a standard concept" in concept_results[("drug-nonstandard", "drug_concept_id")].reason
    assert concept_results[("condition-missing", "condition_concept_id")].status == "review"
    assert "does not exist" in concept_results[("condition-missing", "condition_concept_id")].reason


def test_validation_agent_writes_sql_files_inside_data(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(validation_module, "DEFAULT_DATA_DIR", data_dir)

    sql_output_path = data_dir / "approved.sql"
    review_sql_output_path = data_dir / "review.sql"

    state = run_validation_agent(
        proposed_rows=[
            ProposedInsertRow(
                target_table="observation",
                source_row_id="observations-valid",
                values={
                    "person_id": 1,
                    "observation_concept_id": 3001,
                    "observation_date": "2020-01-12",
                },
                concept_bindings=(ConceptBinding("observation_concept_id", 3001, requires_standard=True),),
                start_date_field="observation_date",
            )
        ],
        sql_output_path=sql_output_path,
        review_sql_output_path=review_sql_output_path,
        execute=True,
        engine=FakeEngine(),
    )

    assert sql_output_path.exists()
    assert review_sql_output_path.exists()
    assert "INSERT INTO [dbo].[observation]" in sql_output_path.read_text(encoding="utf-8")
    assert "-- No rows were routed to the review table." in review_sql_output_path.read_text(encoding="utf-8")

    rendered = render_summary(state)
    assert "Rows approved: 1" in rendered
    assert str(sql_output_path) in rendered


class FakeResult:
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
    concepts = {
        1001: SimpleNamespace(
            concept_id=1001,
            concept_name="Condition standard",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            standard_concept="S",
        ),
        1002: SimpleNamespace(
            concept_id=1002,
            concept_name="Condition source",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            standard_concept=None,
        ),
        2001: SimpleNamespace(
            concept_id=2001,
            concept_name="Wrong domain standard",
            domain_id="Drug",
            vocabulary_id="SNOMED",
            standard_concept="S",
        ),
        3001: SimpleNamespace(
            concept_id=3001,
            concept_name="Observation standard",
            domain_id="Observation",
            vocabulary_id="LOINC",
            standard_concept="S",
        ),
        4001: SimpleNamespace(
            concept_id=4001,
            concept_name="Measurement standard",
            domain_id="Measurement",
            vocabulary_id="LOINC",
            standard_concept="S",
        ),
        5001: SimpleNamespace(
            concept_id=5001,
            concept_name="Drug standard",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            standard_concept="S",
        ),
        5002: SimpleNamespace(
            concept_id=5002,
            concept_name="Drug source nonstandard",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            standard_concept=None,
        ),
    }
    persons = {
        1: date(2000, 1, 1),
    }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}

        if sql.startswith("USE "):
            return FakeResult([])

        if "FROM [dbo].[concept]" in sql and "WHERE concept_id IN" in sql:
            rows = [
                self.concepts[concept_id]
                for concept_id in params.values()
                if concept_id in self.concepts
            ]
            return FakeResult(rows)

        if "FROM [dbo].[person]" in sql and "WHERE person_id IN" in sql:
            rows = [
                SimpleNamespace(person_id=person_id, birth_date=self.persons[person_id])
                for person_id in params.values()
                if person_id in self.persons
            ]
            return FakeResult(rows)

        raise AssertionError(f"Unhandled SQL in fake connection:\n{sql}\nparams={params}")


class FakeEngine:
    def connect(self):
        return FakeConnection()
