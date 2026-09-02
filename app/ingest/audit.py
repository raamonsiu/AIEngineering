"""Turning the catalog into a report a non-engineer can read.

The catalog is the machine-readable artefact; this is its human face. It
exists because the people who most need to know what feeds the system —
a product lead, a legal reviewer, a client asking which of their data is
in there — do not read YAML, and "go look at the repo" is not an answer
to a governance question.

Generated rather than written. A hand-maintained audit document starts
accurate and drifts within weeks, which is worse than none: it is wrong
with authority. Regenerating it on every catalog change (in CI) is what
keeps the audit a living practice instead of a one-off deliverable.
"""

from __future__ import annotations

from app.ingest.catalog import DataCatalog, QualityScore
from app.ingest.orchestrator import IngestionRun

_DIMENSIONS = ("completeness", "consistency", "actuality", "reliability")


def _score_bar(score: int) -> str:
    return "●" * score + "○" * (5 - score)


def generate_audit_report(catalog: DataCatalog, run: IngestionRun | None = None) -> str:
    """Render the catalog (and optionally a pipeline run) as Markdown."""
    included = catalog.included_sources()
    excluded = catalog.excluded_sources()
    review = catalog.sources_under_review()

    lines: list[str] = [
        f"# Reporte de auditoría de datos — {catalog.last_audited}",
        "",
        f"**Fuentes auditadas:** {len(catalog.sources)} · "
        f"**incluidas:** {len(included)} · "
        f"**excluidas:** {len(excluded)} · "
        f"**en revisión:** {len(review)}",
        "",
        "Generado automáticamente desde `data_catalog.yaml`. No editar a mano.",
        "",
        "## Calidad por dimensiones",
        "",
        "| Fuente | Completitud | Consistencia | Actualidad | Fiabilidad | ¿Apta RAG? | Decisión |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in catalog.sources:
        quality = source.quality
        cells = " | ".join(f"{_score_bar(getattr(quality, dim))}" for dim in _DIMENSIONS)
        verdict = "sí" if quality.is_rag_ready else f"**no** ({quality.weakest_dimension})"
        lines.append(f"| `{source.name}` | {cells} | {verdict} | {source.decision.value} |")

    lines += [
        "",
        "> Las dimensiones no se promedian: cada una es condición necesaria. "
        "Una fuente con completitud 5 y fiabilidad 1 no es un 3 — es una fuente "
        "cuyos datos están completos y pueden ser falsos, que es lo peor para RAG.",
        "",
        "## Frescura declarada vs observada",
        "",
        "| Fuente | Declarada | Última actualización | Desfase (días) | Veredicto |",
        "|---|---|---|---:|---|",
    ]
    for source in catalog.sources:
        refresh = source.refresh
        verdict = _staleness_verdict(refresh.declared, refresh.observed_lag_days)
        lines.append(
            f"| `{source.name}` | {refresh.declared} | {refresh.observed_last_update} "
            f"| {refresh.observed_lag_days} | {verdict} |"
        )

    # A source that is included while scoring below the bar is the one
    # combination a reviewer must never have to guess about. Surfaced with
    # its justification rather than left as an apparent contradiction
    # between two tables.
    remediated = [
        s for s in included if not s.quality.is_rag_ready
    ]
    if remediated:
        lines += ["", "## Incluidas pese a no ser aptas tal cual", ""]
        for source in remediated:
            lines.append(
                f"- **`{source.name}`** — dimensión débil: "
                f"**{source.quality.weakest_dimension}** "
                f"({getattr(source.quality, source.quality.weakest_dimension)}/5). "
                f"Remediación declarada: {(source.notes or 'SIN JUSTIFICAR').strip()}"
            )
        lines += [
            "",
            "> La puntuación describe la fuente cruda, no la salida del pipeline. "
            "Incluir una fuente débil es legítimo cuando hay una remediación "
            "declarada para la dimensión débil; sin nota que lo explique, no lo es.",
        ]

    lines += ["", "## Fuentes incluidas", ""]
    for source in included:
        lines.append(
            f"- **`{source.name}`** ({source.format}, {source.volume.records} registros, "
            f"{source.volume.size_mb} MB) — negocio: {source.owner_business}, "
            f"técnico: {source.owner_technical}, origen: `{source.lineage.upstream}`"
        )

    if review:
        lines += ["", "## Fuentes en revisión", ""]
        for source in review:
            lines.append(f"- **`{source.name}`** — {(source.notes or 'sin nota').strip()}")

    if excluded:
        lines += ["", "## Fuentes excluidas deliberadamente", ""]
        for source in excluded:
            lines.append(f"- **`{source.name}`** — {(source.notes or 'ver catálogo').strip()}")
        lines += [
            "",
            "> Excluir es una decisión de higiene, no desidia. Una fuente mala no "
            "produce ruido aleatorio (sería fácil de detectar): produce respuestas "
            "seguras sobre información incorrecta.",
        ]

    lines += ["", "## Datos personales", "", "| Fuente | PII | Tipos | Restricción |", "|---|---|---|---|"]
    for source in catalog.sources:
        sensitivity = source.sensitivity
        lines.append(
            f"| `{source.name}` | {'sí' if sensitivity.contains_pii else 'no'} "
            f"| {', '.join(sensitivity.pii_types) or '—'} "
            f"| {sensitivity.access_restrictions or '—'} |"
        )

    if run is not None:
        lines += _run_section(run)

    return "\n".join(lines) + "\n"


def _staleness_verdict(declared: str, lag_days: int) -> str:
    """Staleness is relative to the declared cadence, never absolute.

    Ten years old on a fifteen-year cycle is fine. Ten years old on a
    yearly cycle is an abandoned source that somebody still believes is
    alive — and that belief is what poisons a corpus.

    Measured in *cycles missed* rather than in days, because a day count
    means nothing without the cadence beside it. One cycle plus a quarter
    of grace is on time; past two and a half, nobody is maintaining it.
    """
    cycle_days = {"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 91, "yearly": 365}
    cycle = cycle_days.get(declared.strip().lower(), 365)
    cycles_missed = lag_days / cycle

    if cycles_missed <= 1.25:
        return "al día"
    if cycles_missed <= 2.5:
        return f"⚠️ con retraso ({cycles_missed:.1f} ciclos)"
    return f"🔴 abandonada ({cycles_missed:.1f} ciclos sin actualizar)"


def _run_section(run: IngestionRun) -> list[str]:
    lines = [
        "",
        "## Última ejecución del pipeline",
        "",
        f"Inicio: {run.started_at.isoformat()} · documentos emitidos: **{run.document_count}**",
        "",
        "| Fuente | Decisión | Ficheros | Unidades | Tras limpieza | Documentos | Válidos | Cuarentena | Descartados | Entidades anonimizadas |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in run.reports:
        lines.append(
            f"| `{report.source_name}` | {report.decision} | {report.files_seen} "
            f"| {report.units_parsed} | {report.units_after_cleaning} "
            f"| {report.documents_emitted} | {report.records_valid} "
            f"| {report.records_quarantined} | {report.records_discarded} "
            f"| {report.entities_anonymized} |"
        )

    divergent = [(r.source_name, r.divergent_duplicates) for r in run.reports if r.divergent_duplicates]
    if divergent:
        lines += ["", "### Duplicados divergentes detectados", ""]
        for name, ids in divergent:
            lines.append(f"- `{name}`: {', '.join(ids)} — resueltos por fecha de firma más reciente")

    errors = [(r.source_name, e) for r in run.reports for e in r.errors]
    if errors:
        lines += ["", "### Errores", ""]
        lines += [f"- `{name}`: {error}" for name, error in errors]

    return lines
