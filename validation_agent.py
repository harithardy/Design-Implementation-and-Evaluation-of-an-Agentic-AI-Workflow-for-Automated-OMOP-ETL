from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

try:
    from src.load_tables import DEFAULT_DATA_DIR, DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier
except ModuleNotFoundError:
    from load_tables import DEFAULT_DATA_DIR, DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier


DEFAULT_TARGET_DATABASE = DEFAULT_DATABASE
DEFAULT_TARGET_SCHEMA = DEFAULT_SCHEMA
DEFAULT_REVIEW_TABLE = "etl_validation_review"
DEFAULT_SQL_OUTPUT_PATH = DEFAULT_DATA_DIR / "validated_insert.sql"
DEFAULT_REVIEW_SQL_OUTPUT_PATH = DEFAULT_DATA_DIR / "validation_review.sql"

TABLE_DOMAIN_RULES = {
    "condition_occurrence": ("condition_concept_id", "Condition"),
    "drug_exposure": ("drug_concept_id", "Drug"),
    "procedure_occurrence": ("procedure_concept_id", "Procedure"),
    "measurement": ("measurement_concept_id", "Measurement"),
    "observation": ("observation_concept_id", "Observation"),
}


@dataclass(frozen=True)
class ConceptBinding:
    field_name: str
    concept_id: int | None
    requires_standard: bool = True
    required_domain: str | None = None


@dataclass
class ProposedInsertRow:
    target_table: str
    source_row_id: str
    values: dict[str, Any]
    concept_bindings: tuple[ConceptBinding, ...] = ()
    start_date_field: str | None = None
    event_date_field: str | None = None
    person_id_field: str | None = "person_id"
    requires_existing_person: bool = True
    uncertainty_reason: str | None = None


@dataclass(frozen=True)
class ConceptReference:
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    standard_concept: str | None


@dataclass(frozen=True)
class ConceptValidationResult:
    source_row_id: str
    target_table: str
    field_name: str
    concept_id: int | None
    status: Literal["validated", "review"]
    reason: str
    concept_name: str | None
    domain_id: str | None
    vocabulary_id: str | None
    standard_concept: str | None


@dataclass(frozen=True)
class RowValidationResult:
    source_row_id: str
    target_table: str
    status: Literal["approved", "review"]
    reason: str


class ValidationAgentState(TypedDict, total=False):
    proposal_file: Path
    sql_output_path: Path
    review_sql_output_path: Path
    target_database: str
    schema: str
    review_table: str
    execute: bool
    engine: Any
    proposed_rows: list[ProposedInsertRow]
    concept_references: dict[int, ConceptReference]
    person_birth_dates: dict[int, date | None]
    concept_results: list[ConceptValidationResult]
    concept_issues_by_row: dict[str, list[str]]
    row_results: list[RowValidationResult]
    review_reasons: dict[str, str]
    approved_rows: list[ProposedInsertRow]
    review_rows: list[ProposedInsertRow]
    generated_sql: dict[str, str]
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


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, datetime):
        return sql_unicode_literal(value.isoformat(sep=" ", timespec="seconds"))
    if isinstance(value, date):
        return sql_unicode_literal(value.isoformat())
    return sql_unicode_literal(str(value))


def serialize_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return value


def review_payload(row: ProposedInsertRow) -> str:
    payload = {
        "target_table": row.target_table,
        "source_row_id": row.source_row_id,
        "values": {key: serialize_json_value(value) for key, value in row.values.items()},
        "concept_bindings": [asdict(binding) for binding in row.concept_bindings],
        "start_date_field": row.start_date_field,
        "event_date_field": row.event_date_field,
        "person_id_field": row.person_id_field,
        "requires_existing_person": row.requires_existing_person,
        "uncertainty_reason": row.uncertainty_reason,
    }
    return json.dumps(payload, default=serialize_json_value, sort_keys=True)


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values cannot be coerced to integer identifiers.")
    if isinstance(value, int):
        return value
    text_value = str(value).strip()
    if text_value == "":
        return None
    return int(text_value)


def coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text_value = str(value).strip()
    if not text_value:
        return None

    normalized = text_value.replace("Z", "+00:00")
    try:
        if "T" in normalized or " " in normalized:
            return datetime.fromisoformat(normalized).date()
        return date.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{text_value!r} is not a valid ISO date or datetime.") from error


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    data_root = DEFAULT_DATA_DIR.resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"Output path must be inside {data_root}.") from error
    if resolved.suffix.lower() != ".sql":
        raise ValueError("Output path must use the .sql extension.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def coerce_binding(raw_binding: ConceptBinding | dict[str, Any]) -> ConceptBinding:
    if isinstance(raw_binding, ConceptBinding):
        return raw_binding
    return ConceptBinding(
        field_name=str(raw_binding["field_name"]),
        concept_id=coerce_int(raw_binding.get("concept_id")),
        requires_standard=bool(raw_binding.get("requires_standard", True)),
        required_domain=(
            str(raw_binding["required_domain"])
            if raw_binding.get("required_domain") is not None
            else None
        ),
    )


def coerce_proposed_row(raw_row: ProposedInsertRow | dict[str, Any]) -> ProposedInsertRow:
    if isinstance(raw_row, ProposedInsertRow):
        return raw_row

    values = dict(raw_row.get("values", {}))
    concept_bindings = tuple(
        coerce_binding(raw_binding)
        for raw_binding in raw_row.get("concept_bindings", [])
    )
    return ProposedInsertRow(
        target_table=str(raw_row["target_table"]),
        source_row_id=str(raw_row["source_row_id"]),
        values=values,
        concept_bindings=concept_bindings,
        start_date_field=(
            str(raw_row["start_date_field"])
            if raw_row.get("start_date_field") is not None
            else None
        ),
        event_date_field=(
            str(raw_row["event_date_field"])
            if raw_row.get("event_date_field") is not None
            else None
        ),
        person_id_field=(
            str(raw_row["person_id_field"])
            if raw_row.get("person_id_field") is not None
            else None
        ),
        requires_existing_person=bool(raw_row.get("requires_existing_person", True)),
        uncertainty_reason=(
            str(raw_row["uncertainty_reason"])
            if raw_row.get("uncertainty_reason") is not None
            else None
        ),
    )


def load_proposed_rows(state: ValidationAgentState) -> ValidationAgentState:
    raw_rows = state.get("proposed_rows")
    if raw_rows:
        proposed_rows = [coerce_proposed_row(raw_row) for raw_row in raw_rows]
        return {"proposed_rows": proposed_rows}

    proposal_file = state.get("proposal_file")
    if proposal_file is None:
        raise ValueError("Provide proposed_rows directly or set proposal_file.")

    file_path = Path(proposal_file)
    raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("Proposal file must contain a JSON array of proposed rows.")
    proposed_rows = [coerce_proposed_row(raw_row) for raw_row in raw_payload]
    if not proposed_rows:
        raise ValueError("No proposed rows were supplied for validation.")
    return {"proposed_rows": proposed_rows}


def fetch_concept_references(connection: Any, concept_ids: list[int]) -> dict[int, ConceptReference]:
    references: dict[int, ConceptReference] = {}
    if not concept_ids:
        return references

    batch_size = 500
    for start in range(0, len(concept_ids), batch_size):
        batch = concept_ids[start : start + batch_size]
        placeholders = ", ".join(f":concept_{index}" for index, _ in enumerate(batch))
        params = {f"concept_{index}": concept_id for index, concept_id in enumerate(batch)}
        rows = connection.execute(
            text(
                f"""
                SELECT concept_id, concept_name, domain_id, vocabulary_id, standard_concept
                FROM [dbo].[concept]
                WHERE concept_id IN ({placeholders})
                """
            ),
            params,
        )
        for row in rows:
            references[int(row.concept_id)] = ConceptReference(
                concept_id=int(row.concept_id),
                concept_name=row.concept_name,
                domain_id=row.domain_id,
                vocabulary_id=row.vocabulary_id,
                standard_concept=row.standard_concept,
            )
    return references


def fetch_person_birth_dates(connection: Any, person_ids: list[int], schema_name: str) -> dict[int, date | None]:
    birth_dates: dict[int, date | None] = {}
    if not person_ids:
        return birth_dates

    batch_size = 500
    for start in range(0, len(person_ids), batch_size):
        batch = person_ids[start : start + batch_size]
        placeholders = ", ".join(f":person_{index}" for index, _ in enumerate(batch))
        params = {f"person_{index}": person_id for index, person_id in enumerate(batch)}
        rows = connection.execute(
            text(
                f"""
                SELECT
                    person_id,
                    COALESCE(
                        CAST(birth_datetime AS date),
                        DATEFROMPARTS(
                            year_of_birth,
                            COALESCE(month_of_birth, 1),
                            COALESCE(day_of_birth, 1)
                        )
                    ) AS birth_date
                FROM {build_table_name(schema_name, 'person')}
                WHERE person_id IN ({placeholders})
                """
            ),
            params,
        )
        for row in rows:
            birth_dates[int(row.person_id)] = coerce_date(row.birth_date)
    return birth_dates


def load_reference_data(state: ValidationAgentState) -> ValidationAgentState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_TARGET_DATABASE)
    schema_name = state.get("schema", DEFAULT_TARGET_SCHEMA)

    concept_ids = sorted(
        {
            binding.concept_id
            for row in state["proposed_rows"]
            for binding in row.concept_bindings
            if binding.concept_id is not None
        }
    )
    person_ids = sorted(
        {
            person_id
            for row in state["proposed_rows"]
            if row.requires_existing_person and row.person_id_field
            for person_id in [coerce_int(row.values.get(row.person_id_field))]
            if person_id is not None
        }
    )

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))
        concept_references = fetch_concept_references(connection, concept_ids)
        person_birth_dates = fetch_person_birth_dates(connection, person_ids, schema_name)

    return {
        "concept_references": concept_references,
        "person_birth_dates": person_birth_dates,
    }


def expected_domain(target_table: str, binding: ConceptBinding) -> str | None:
    table_rule = TABLE_DOMAIN_RULES.get(target_table)
    if table_rule and binding.field_name == table_rule[0]:
        return table_rule[1]
    return binding.required_domain


def requires_standard(target_table: str, binding: ConceptBinding) -> bool:
    table_rule = TABLE_DOMAIN_RULES.get(target_table)
    if table_rule and binding.field_name == table_rule[0]:
        return True
    return binding.requires_standard


def validate_concepts(state: ValidationAgentState) -> ValidationAgentState:
    concept_references = state.get("concept_references", {})
    concept_results: list[ConceptValidationResult] = []
    concept_issues_by_row: dict[str, list[str]] = {}

    for row in state["proposed_rows"]:
        issues = concept_issues_by_row.setdefault(row.source_row_id, [])
        bound_fields = {binding.field_name for binding in row.concept_bindings}

        for field_name in sorted(
            field_name
            for field_name in row.values
            if field_name.endswith("_concept_id") and field_name not in bound_fields
        ):
            reason = f"Missing concept binding metadata for {field_name}."
            issues.append(reason)
            concept_results.append(
                ConceptValidationResult(
                    source_row_id=row.source_row_id,
                    target_table=row.target_table,
                    field_name=field_name,
                    concept_id=coerce_int(row.values.get(field_name)),
                    status="review",
                    reason=reason,
                    concept_name=None,
                    domain_id=None,
                    vocabulary_id=None,
                    standard_concept=None,
                )
            )

        primary_rule = TABLE_DOMAIN_RULES.get(row.target_table)
        if primary_rule and primary_rule[0] not in bound_fields:
            reason = f"Missing binding for required primary concept field {primary_rule[0]}."
            issues.append(reason)
            concept_results.append(
                ConceptValidationResult(
                    source_row_id=row.source_row_id,
                    target_table=row.target_table,
                    field_name=primary_rule[0],
                    concept_id=None,
                    status="review",
                    reason=reason,
                    concept_name=None,
                    domain_id=None,
                    vocabulary_id=None,
                    standard_concept=None,
                )
            )

        for binding in row.concept_bindings:
            field_value = row.values.get(binding.field_name)
            try:
                field_concept_id = coerce_int(field_value)
            except ValueError:
                field_concept_id = None

            if binding.field_name not in row.values:
                reason = f"Field {binding.field_name} is missing from the proposed values."
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=None,
                        domain_id=None,
                        vocabulary_id=None,
                        standard_concept=None,
                    )
                )
                continue

            if binding.concept_id is None or field_concept_id is None:
                reason = f"{binding.field_name} concept_id must not be null."
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=None,
                        domain_id=None,
                        vocabulary_id=None,
                        standard_concept=None,
                    )
                )
                continue

            if field_concept_id != binding.concept_id:
                reason = (
                    f"Binding mismatch for {binding.field_name}: values contains {field_concept_id}, "
                    f"binding contains {binding.concept_id}."
                )
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=None,
                        domain_id=None,
                        vocabulary_id=None,
                        standard_concept=None,
                    )
                )
                continue

            concept_reference = concept_references.get(binding.concept_id)
            if concept_reference is None:
                reason = f"concept_id {binding.concept_id} does not exist in dbo.concept."
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=None,
                        domain_id=None,
                        vocabulary_id=None,
                        standard_concept=None,
                    )
                )
                continue

            required_domain = expected_domain(row.target_table, binding)
            if requires_standard(row.target_table, binding) and concept_reference.standard_concept != "S":
                reason = (
                    f"concept_id {binding.concept_id} is not a standard concept for "
                    f"{binding.field_name}."
                )
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=concept_reference.concept_name,
                        domain_id=concept_reference.domain_id,
                        vocabulary_id=concept_reference.vocabulary_id,
                        standard_concept=concept_reference.standard_concept,
                    )
                )
                continue

            if required_domain and concept_reference.domain_id != required_domain:
                reason = (
                    f"concept_id {binding.concept_id} belongs to domain "
                    f"{concept_reference.domain_id}, expected {required_domain}."
                )
                issues.append(reason)
                concept_results.append(
                    ConceptValidationResult(
                        source_row_id=row.source_row_id,
                        target_table=row.target_table,
                        field_name=binding.field_name,
                        concept_id=binding.concept_id,
                        status="review",
                        reason=reason,
                        concept_name=concept_reference.concept_name,
                        domain_id=concept_reference.domain_id,
                        vocabulary_id=concept_reference.vocabulary_id,
                        standard_concept=concept_reference.standard_concept,
                    )
                )
                continue

            concept_results.append(
                ConceptValidationResult(
                    source_row_id=row.source_row_id,
                    target_table=row.target_table,
                    field_name=binding.field_name,
                    concept_id=binding.concept_id,
                    status="validated",
                    reason="Concept binding validated.",
                    concept_name=concept_reference.concept_name,
                    domain_id=concept_reference.domain_id,
                    vocabulary_id=concept_reference.vocabulary_id,
                    standard_concept=concept_reference.standard_concept,
                )
            )

    return {
        "concept_results": concept_results,
        "concept_issues_by_row": concept_issues_by_row,
    }


def extract_person_id(row: ProposedInsertRow) -> int | None:
    if row.person_id_field is None:
        return None
    return coerce_int(row.values.get(row.person_id_field))


def validate_rows(state: ValidationAgentState) -> ValidationAgentState:
    person_birth_dates = state.get("person_birth_dates", {})
    concept_issues_by_row = state.get("concept_issues_by_row", {})
    today = date.today()

    row_results: list[RowValidationResult] = []
    review_reasons: dict[str, str] = {}
    approved_rows: list[ProposedInsertRow] = []
    review_rows: list[ProposedInsertRow] = []

    for row in state["proposed_rows"]:
        issues = list(concept_issues_by_row.get(row.source_row_id, []))

        if row.uncertainty_reason:
            issues.append(f"Uncertain mapping: {row.uncertainty_reason}")

        person_id = extract_person_id(row)
        if row.requires_existing_person:
            if person_id is None:
                issues.append(f"{row.person_id_field or 'person_id'} must not be null.")
            elif person_id not in person_birth_dates:
                issues.append(f"person_id {person_id} does not exist in OMOP person.")

        event_date: date | None = None
        if row.start_date_field:
            try:
                start_date = coerce_date(row.values.get(row.start_date_field))
            except ValueError:
                start_date = None
                issues.append(f"{row.start_date_field} is not a valid date.")
            if start_date is None:
                issues.append(f"{row.start_date_field} must not be null.")

        event_date_field = row.event_date_field or row.start_date_field
        if event_date_field:
            raw_event_date = row.values.get(event_date_field)
            if raw_event_date not in (None, ""):
                try:
                    event_date = coerce_date(raw_event_date)
                except ValueError:
                    issues.append(f"{event_date_field} is not a valid date.")

        if person_id is not None and person_id in person_birth_dates and event_date is not None:
            birth_date = person_birth_dates.get(person_id)
            if birth_date is None:
                issues.append(f"Birth date for person_id {person_id} is unavailable.")
            elif event_date < birth_date:
                issues.append(
                    f"{event_date_field or 'event_date'} {event_date.isoformat()} occurs before "
                    f"birth date {birth_date.isoformat()}."
                )

        if event_date is not None and event_date > today:
            issues.append(
                f"{event_date_field or 'event_date'} {event_date.isoformat()} is in the future."
            )

        if issues:
            reason = "; ".join(dict.fromkeys(issues))
            review_rows.append(row)
            review_reasons[row.source_row_id] = reason
            row_results.append(
                RowValidationResult(
                    source_row_id=row.source_row_id,
                    target_table=row.target_table,
                    status="review",
                    reason=reason,
                )
            )
            continue

        approved_rows.append(row)
        row_results.append(
            RowValidationResult(
                source_row_id=row.source_row_id,
                target_table=row.target_table,
                status="approved",
                reason="Row passed validation and can be rendered into SQL.",
            )
        )

    return {
        "row_results": row_results,
        "review_reasons": review_reasons,
        "approved_rows": approved_rows,
        "review_rows": review_rows,
    }


def render_insert_statement(row: ProposedInsertRow, schema_name: str) -> str:
    columns = ", ".join(quote_identifier(column_name) for column_name in row.values)
    values = ", ".join(sql_literal(value) for value in row.values.values())
    return (
        f"INSERT INTO {build_table_name(schema_name, row.target_table)} ({columns}) "
        f"VALUES ({values});"
    )


def build_review_table_statement(schema_name: str, review_table: str) -> str:
    qualified_name = build_table_name(schema_name, review_table)
    return "\n".join(
        [
            f"IF OBJECT_ID(N'{qualified_name}', N'U') IS NULL",
            "BEGIN",
            f"    CREATE TABLE {qualified_name} (",
            "        [review_id] int IDENTITY(1,1) NOT NULL PRIMARY KEY,",
            "        [target_table] varchar(128) NOT NULL,",
            "        [source_row_id] varchar(128) NOT NULL,",
            "        [review_status] varchar(32) NOT NULL,",
            "        [review_reason] nvarchar(max) NOT NULL,",
            "        [payload_json] nvarchar(max) NOT NULL,",
            "        [created_at_utc] datetime2 NOT NULL CONSTRAINT [DF_etl_validation_review_created_at_utc] DEFAULT SYSUTCDATETIME()",
            "    )",
            "END;",
        ]
    )


def build_review_insert_statement(
    row: ProposedInsertRow,
    schema_name: str,
    review_table: str,
    reason: str,
) -> str:
    qualified_name = build_table_name(schema_name, review_table)
    payload_json = review_payload(row)
    return (
        f"INSERT INTO {qualified_name} "
        f"([target_table], [source_row_id], [review_status], [review_reason], [payload_json]) "
        f"VALUES ("
        f"{sql_unicode_literal(row.target_table)}, "
        f"{sql_unicode_literal(row.source_row_id)}, "
        f"{sql_unicode_literal('review')}, "
        f"{sql_unicode_literal(reason)}, "
        f"{sql_unicode_literal(payload_json)}"
        f");"
    )


def generate_sql(state: ValidationAgentState) -> ValidationAgentState:
    schema_name = state.get("schema", DEFAULT_TARGET_SCHEMA)
    review_table = state.get("review_table", DEFAULT_REVIEW_TABLE)
    review_reasons = state.get("review_reasons", {})

    approved_rows = state.get("approved_rows", [])
    review_rows = state.get("review_rows", [])

    approved_sql_lines = ["SET NOCOUNT ON;"]
    if approved_rows:
        approved_sql_lines.extend(
            render_insert_statement(row, schema_name)
            for row in approved_rows
        )
    else:
        approved_sql_lines.append("-- No approved rows passed validation.")

    review_sql_lines = ["SET NOCOUNT ON;"]
    if review_rows:
        review_sql_lines.append(build_review_table_statement(schema_name, review_table))
        review_sql_lines.extend(
            build_review_insert_statement(
                row=row,
                schema_name=schema_name,
                review_table=review_table,
                reason=review_reasons[row.source_row_id],
            )
            for row in review_rows
        )
    else:
        review_sql_lines.append("-- No rows were routed to the review table.")

    return {
        "generated_sql": {
            "approved_sql": "\n\n".join(approved_sql_lines),
            "review_sql": "\n\n".join(review_sql_lines),
        }
    }


def should_write_outputs(state: ValidationAgentState) -> str:
    return "write" if state.get("execute", True) else "summarize"


def write_sql_files(state: ValidationAgentState) -> ValidationAgentState:
    approved_path = validate_output_path(Path(state.get("sql_output_path", DEFAULT_SQL_OUTPUT_PATH)))
    review_path = validate_output_path(
        Path(state.get("review_sql_output_path", DEFAULT_REVIEW_SQL_OUTPUT_PATH))
    )
    generated_sql = state.get("generated_sql", {})

    approved_path.write_text(generated_sql.get("approved_sql", ""), encoding="utf-8")
    review_path.write_text(generated_sql.get("review_sql", ""), encoding="utf-8")
    return {
        "sql_output_path": approved_path,
        "review_sql_output_path": review_path,
    }


def summarize_run(state: ValidationAgentState) -> ValidationAgentState:
    concept_results = state.get("concept_results", [])
    row_results = state.get("row_results", [])
    approved_rows = state.get("approved_rows", [])
    review_rows = state.get("review_rows", [])
    generated_sql = state.get("generated_sql", {})
    return {
        "summary": {
            "rows_proposed": len(state.get("proposed_rows", [])),
            "concept_bindings_checked": len(concept_results),
            "concept_bindings_validated": sum(
                1 for result in concept_results if result.status == "validated"
            ),
            "rows_approved": len(approved_rows),
            "rows_reviewed": len(review_rows),
            "approved_statement_count": len(approved_rows),
            "review_statement_count": len(review_rows),
            "approved_row_ids": [result.source_row_id for result in row_results if result.status == "approved"],
            "review_row_ids": [result.source_row_id for result in row_results if result.status == "review"],
            "sql_output_path": str(state.get("sql_output_path", DEFAULT_SQL_OUTPUT_PATH)),
            "review_sql_output_path": str(
                state.get("review_sql_output_path", DEFAULT_REVIEW_SQL_OUTPUT_PATH)
            ),
            "approved_sql_generated": bool(generated_sql.get("approved_sql")),
            "review_sql_generated": bool(generated_sql.get("review_sql")),
        }
    }


def build_validation_graph():
    workflow = StateGraph(ValidationAgentState)
    workflow.add_node("load_rows", load_proposed_rows)
    workflow.add_node("load_refs", load_reference_data)
    workflow.add_node("validate_concepts", validate_concepts)
    workflow.add_node("validate_rows", validate_rows)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("write", write_sql_files)
    workflow.add_node("summarize", summarize_run)

    workflow.set_entry_point("load_rows")
    workflow.add_edge("load_rows", "load_refs")
    workflow.add_edge("load_refs", "validate_concepts")
    workflow.add_edge("validate_concepts", "validate_rows")
    workflow.add_edge("validate_rows", "generate_sql")
    workflow.add_conditional_edges(
        "generate_sql",
        should_write_outputs,
        {"write": "write", "summarize": "summarize"},
    )
    workflow.add_edge("write", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


validation_agent = build_validation_graph()


def run_validation_agent(
    proposal_file: Path | None = None,
    proposed_rows: list[ProposedInsertRow] | list[dict[str, Any]] | None = None,
    target_database: str = DEFAULT_TARGET_DATABASE,
    schema: str = DEFAULT_TARGET_SCHEMA,
    review_table: str = DEFAULT_REVIEW_TABLE,
    sql_output_path: Path = DEFAULT_SQL_OUTPUT_PATH,
    review_sql_output_path: Path = DEFAULT_REVIEW_SQL_OUTPUT_PATH,
    execute: bool = True,
    engine: Any | None = None,
) -> ValidationAgentState:
    initial_state: ValidationAgentState = {
        "target_database": target_database,
        "schema": schema,
        "review_table": review_table,
        "sql_output_path": Path(sql_output_path),
        "review_sql_output_path": Path(review_sql_output_path),
        "execute": execute,
    }
    if proposal_file is not None:
        initial_state["proposal_file"] = Path(proposal_file)
    if proposed_rows is not None:
        initial_state["proposed_rows"] = [coerce_proposed_row(raw_row) for raw_row in proposed_rows]
    if engine is not None:
        initial_state["engine"] = engine
    return validation_agent.invoke(initial_state)


def render_summary(state: ValidationAgentState) -> str:
    summary = state.get("summary", {})
    return "\n".join(
        [
            f"Rows proposed: {summary.get('rows_proposed', 0)}",
            f"Concept bindings checked: {summary.get('concept_bindings_checked', 0)}",
            f"Concept bindings validated: {summary.get('concept_bindings_validated', 0)}",
            f"Rows approved: {summary.get('rows_approved', 0)}",
            f"Rows reviewed: {summary.get('rows_reviewed', 0)}",
            f"Approved statement count: {summary.get('approved_statement_count', 0)}",
            f"Review statement count: {summary.get('review_statement_count', 0)}",
            (
                "Approved row ids: " + ", ".join(summary.get("approved_row_ids", []))
                if summary.get("approved_row_ids")
                else "Approved row ids: none"
            ),
            (
                "Review row ids: " + ", ".join(summary.get("review_row_ids", []))
                if summary.get("review_row_ids")
                else "Review row ids: none"
            ),
            f"SQL output path: {summary.get('sql_output_path', DEFAULT_SQL_OUTPUT_PATH)}",
            f"Review SQL output path: {summary.get('review_sql_output_path', DEFAULT_REVIEW_SQL_OUTPUT_PATH)}",
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate proposed OMOP insert rows against dbo.concept and dbo.person before "
            "rendering final SQL. Invalid or uncertain rows are routed to a review table SQL plan."
        )
    )
    parser.add_argument(
        "--proposal-file",
        type=Path,
        required=True,
        help="JSON file containing the proposed OMOP rows to validate.",
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
        "--sql-output",
        type=Path,
        default=DEFAULT_SQL_OUTPUT_PATH,
        help="Output SQL file for approved rows. Must be inside data/.",
    )
    parser.add_argument(
        "--review-sql-output",
        type=Path,
        default=DEFAULT_REVIEW_SQL_OUTPUT_PATH,
        help="Output SQL file for review-table rows. Must be inside data/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and generate SQL in memory without writing the output files.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_validation_agent(
        proposal_file=args.proposal_file,
        target_database=args.database,
        schema=args.schema,
        review_table=args.review_table,
        sql_output_path=args.sql_output,
        review_sql_output_path=args.review_sql_output,
        execute=not args.dry_run,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
