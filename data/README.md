# `data/` — synthetic enterprise corpus

Everything under `data/corpus/` is **generated**, fake and safe to commit. No
real client, person, email, phone number or amount appears here.

The corpus exists so the ingestion/cleaning work can be tested against data
that is deliberately as messy as the real thing, with the mess written down.
It is coherent with the world of `app/context/examples.py`: a Spanish software
consultancy, EUR, hours billed at **62.50 EUR/hour**, phases and
tasks, teams and durations in weeks.

## Regenerate

```bash
uv run python data/build_corpus.py
```

The generator is deterministic: fixed seed (`SEED = 20240115`), fixed document
timestamps, no `datetime.now()` anywhere in the content. Two consecutive runs
produce **byte-identical files**, so re-running it on a clean checkout leaves an
empty diff. `data/corpus/` is wiped and rebuilt on every run, so it is also
idempotent in the stronger sense: nothing stale survives.

Two extra steps are needed for that guarantee and are easy to break by accident:
PyMuPDF stamps a random `/ID` into each PDF trailer, and ZIP containers (docx,
xlsx) record the wall clock per entry. Both are frozen after the library writes
the bytes.

## Contents

```
data/
├── build_corpus.py
├── README.md
└── corpus/
    ├── historical_budgets/    28 × *.json     31.9 KB
    ├── meeting_transcripts/   12 × *.txt      32.6 KB
    ├── proposal_templates/     5 × *.docx    186.2 KB
    ├── signed_contracts/       6 × *.pdf      56.3 KB
    └── rate_card/              1 × *.xlsx      5.8 KB
```

Total: **312.8 KB**.

| Source | What it is |
|---|---|
| `historical_budgets/` | 28 JSON files, one budget per file, as exported from the ERP. Fields: `budget_id`, `client_name`, `client_code`, `project_type`, `total_amount`, `currency`, `hours_estimated`, `signed_at`, `status`, `contact_email`, `contact_phone`, `account_manager`, `phases[]` (`name`, `hours`, `cost_eur`). Canonical forms: `BUDGET-YYYY-NNNN`, `CLI-NNNN`, status ∈ {draft, signed, rejected}. |
| `meeting_transcripts/` | 12 Spanish transcripts of client scoping/kickoff meetings, in **two format eras**. 2024 and later (7 files): every line is `[hh:mm:ss] Nombre Apellido: texto`. Before 2024 (5 files): no speaker tags, free-running paragraphs, irregular blank lines and double spaces. All of them carry inline PII — Spanish names, organisations, emails, phone numbers — which is what an anonymisation step has to catch. |
| `proposal_templates/` | 5 Spanish Word proposals with a real `add_heading` hierarchy (level 0 title, level 1 sections `Alcance` / `Entregables` / `Cronograma` / `Equipo` / `Condiciones económicas`, level 2 subsections) and a real table of phases, hours and amounts. A parser that splits by heading style will find them. |
| `signed_contracts/` | 6 Spanish services contracts as **born-digital** PDFs, 1–3 pages, with a genuine text layer. No scans, no image-only pages, no OCR required. Each references a `budget_id` and a client. |
| `rate_card/` | 1 XLSX, role × seniority, rates centred on 62.50 EUR/h. **Deliberately stale**: its mtime is set to 2024-01-18 while every other corpus file is stamped 2026-09-01, so a staleness rule reading the filesystem will exclude it. |

## Planted defects — ground truth

18 of the 28 budget files (64 %) are completely clean: canonical
`budget_id` and `client_code`, ISO dates, `EUR`, numeric amounts, no sentinels,
and `total_amount == hours_estimated × 62.50` with `phases[]` summing to both.
The table below is exhaustive: every defect in the corpus is listed here, and
every row below is actually present on disk — the generator applies these rows
rather than describing them.

| File (under `data/corpus/`) | Field | Family | Planted value | Notes |
|---|---|---|---|---|
| `historical_budgets/budget_019.json` | `client_name` | F1 Heterogeneidad de formato | `'Acme Corp'` | Variante ortográfica de «ACME Corp.» (canónico en budget_003.json). |
| `historical_budgets/budget_019.json` | `signed_at` | F1 Heterogeneidad de formato | `'15/03/2024'` | Fecha en formato DD/MM/YYYY; misma fecha que budget_003.json («2024-03-15»). |
| `historical_budgets/budget_019.json` | `currency` | F1 Heterogeneidad de formato | `'eur'` | Moneda en minúsculas. |
| `historical_budgets/budget_019.json` | `total_amount` | F1 Heterogeneidad de formato | `'80000'` | Importe como string sin separadores (valor real 80000.0). |
| `historical_budgets/budget_020.json` | `client_name` | F1 Heterogeneidad de formato | `'acme corp'` | Variante ortográfica en minúsculas de «ACME Corp.». |
| `historical_budgets/budget_020.json` | `client_code` | F1 Heterogeneidad de formato | `'cli-1001'` | Código de cliente en minúsculas (canónico «CLI-1001»). |
| `historical_budgets/budget_020.json` | `signed_at` | F1 Heterogeneidad de formato | `'Mar 15 2024'` | Fecha en formato «Mon DD YYYY» en inglés; misma fecha que budget_003.json. |
| `historical_budgets/budget_020.json` | `currency` | F1 Heterogeneidad de formato | `'€'` | Moneda como símbolo. |
| `historical_budgets/budget_020.json` | `total_amount` | F1 Heterogeneidad de formato | `'80.000,00'` | Importe como string con separador de miles y decimal europeos (valor real 80000.0). |
| `historical_budgets/budget_021.json` | `client_name` | F1 Heterogeneidad de formato | `'ACME'` | Variante abreviada de «ACME Corp.». |
| `historical_budgets/budget_021.json` | `currency` | F1 Heterogeneidad de formato | `'euros'` | Moneda escrita en palabra. |
| `historical_budgets/budget_021.json` | `total_amount` | F1 Heterogeneidad de formato | `'56250'` | Importe como string sin separadores (valor real 56250.0). |
| `meeting_transcripts/2024-05-02_grupo-navarro_revision-presupuesto.txt` | `(cuerpo del acta)` | F1 Heterogeneidad de formato | — | El importe aparece como «52.500,00 €», como «52500 EUR» y escrito en letra; la fecha aparece como «02/05/2024» y como «2 de mayo». |
| `meeting_transcripts/2023-02-27_grupo-navarro_seguimiento.txt` | `(cuerpo del acta)` | F1 Heterogeneidad de formato | — | El mismo cliente se nombra «Grupo Navarro Logística», «grupo navarro» y «GN Logistica»; la moneda aparece como «euros» y como «EUR». |
| `historical_budgets/budget_023.json` | `budget_id` | F2 Duplicados divergentes | `'BUDGET-2024-0312'` | Mismo budget_id que budget_022.json. Importes divergentes: 48750.0 (022) vs 52500.0 (023); signed_at 2024-03-12 (022) vs 2024-05-02 (023), así que «quedarse con el más reciente» resuelve a budget_023.json. |
| `historical_budgets/budget_025.json` | `budget_id` | F2 Duplicados divergentes | `'BUDGET-2023-0188'` | Mismo budget_id que budget_024.json. Importes divergentes: 31250.0 (024) vs 29875.0 (025); signed_at 2023-06-04 (024) vs 2023-09-18 (025), así que «quedarse con el más reciente» resuelve a budget_025.json. |
| `meeting_transcripts/2024-05-02_grupo-navarro_revision-presupuesto.txt` | `(cuerpo del acta)` | F2 Duplicados divergentes | — | El acta cita BUDGET-2024-0312 por 52.500,00 €, que contradice budget_022.json (48750.0) y coincide con budget_023.json (52500.0). |
| `historical_budgets/budget_026.json` | `client_name` | F3 Nulos disfrazados | `'N/A'` | Centinela «N/A» en lugar del nombre del cliente (es CLI-1004, Farmacias Delgado). |
| `historical_budgets/budget_026.json` | `contact_email` | F3 Nulos disfrazados | `'unknown'` | Centinela «unknown» en lugar de un correo. |
| `historical_budgets/budget_026.json` | `account_manager` | F3 Nulos disfrazados | `'TBD'` | Centinela «TBD» en lugar del responsable de cuenta. |
| `historical_budgets/budget_027.json` | `client_name` | F3 Nulos disfrazados | `'-'` | Centinela «-» en lugar del nombre del cliente (es CLI-1007, Inmobiliaria Peñalver). |
| `historical_budgets/budget_027.json` | `contact_email` | F3 Nulos disfrazados | `''` | Cadena vacía. |
| `historical_budgets/budget_027.json` | `account_manager` | F3 Nulos disfrazados | `'pendiente'` | Centinela «pendiente» en lugar del responsable de cuenta. |
| `historical_budgets/budget_028.json` | `client_name` | F3 Nulos disfrazados | `' '` | Cadena de un solo espacio: parece rellena hasta que se hace strip() (es CLI-1005, Bodegas Villanueva). |
| `meeting_transcripts/2023-02-27_grupo-navarro_seguimiento.txt` | `(bloque de asistentes)` | F3 Nulos disfrazados | — | La cabecera del acta trae «Responsable de cuenta: pendiente», «Correo de contacto: N/A» y «Teléfono: -». |
| `historical_budgets/budget_026.json` | `total_amount` | F4 Fuera de rango | `-18750.0` | Importe negativo. Además ya no cuadra con phases[] ni con hours_estimated * 62.50 (= 18750.0). |
| `historical_budgets/budget_027.json` | `total_amount` | F4 Fuera de rango | `99000000` | Importe absurdo (99 millones) frente a las 240 h del propio registro (= 15000.0). |
| `historical_budgets/budget_028.json` | `hours_estimated` | F4 Fuera de rango | `1480000` | Horas en millones; phases[] sigue sumando 260 h. |
| `historical_budgets/budget_028.json` | `signed_at` | F4 Fuera de rango | `'2031-07-14'` | Fecha de firma en el futuro. |
| `historical_budgets/budget_028.json` | `budget_id` | F4 Fuera de rango | `'BDG_2024_12'` | budget_id fuera del patrón canónico BUDGET-YYYY-NNNN. |

### Count per family

| Family | Planted instances |
|---|---|
| F1 Heterogeneidad de formato | 14 |
| F2 Duplicados divergentes | 3 |
| F3 Nulos disfrazados | 8 |
| F4 Fuera de rango | 5 |

### Reading the families

- **F1 Heterogeneidad de formato** — the same fact written several ways. Dates in three formats
  (`2024-03-15` in the clean majority, `15/03/2024`, `Mar 15 2024` — all three
  denote the same day). Currency as `EUR`, `eur`, `€`, `euros`. `ACME Corp.`
  (canonical, `budget_003.json`) also appearing as `Acme Corp`, `acme corp` and
  `ACME`. `total_amount` as a number in the clean files, as `"80000"` and as
  `"80.000,00"` in the heterogeneous ones.
- **F2 Duplicados divergentes** — two `budget_id` values appear in two files each with different
  `total_amount` *and* different `signed_at`, so a "keep the latest" rule has
  something to resolve: `BUDGET-2024-0312` (`budget_022.json` 48750.0 @ 2024-03-12
  vs `budget_023.json` 52500.0 @ 2024-05-02) and `BUDGET-2023-0188`
  (`budget_024.json` 31250.0 @ 2023-06-04 vs `budget_025.json` 29875.0 @ 2023-09-18).
  `budget_022.json` and `budget_024.json` carry no defect of their own and are
  listed here only as the counterparts of the collision — but because they
  participate in one, they are not counted in the clean majority above.
- **F3 Nulos disfrazados** — all seven sentinels appear: `"N/A"`, `"-"`, `"unknown"`, `"TBD"`,
  `"pendiente"`, `""` and `" "`, spread across `client_name`, `contact_email`
  and `account_manager`, plus a transcript whose attendee block is filled with
  the same placeholders.
- **F4 Fuera de rango** — a negative `total_amount`, an absurd one (99 000 000), a
  `hours_estimated` in the millions, a `signed_at` in the future, and one
  `budget_id` outside the canonical pattern.

## What is *not* in here

No parser, no FastAPI endpoint, no validation logic. This directory only
produces data; the code that consumes it lives elsewhere.
