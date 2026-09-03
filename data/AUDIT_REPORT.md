# Reporte de auditoría de datos — 2026-09-02

**Fuentes auditadas:** 5 · **incluidas:** 3 · **excluidas:** 1 · **en revisión:** 1

Generado automáticamente desde `data_catalog.yaml`. No editar a mano.

## Calidad por dimensiones

| Fuente | Completitud | Consistencia | Actualidad | Fiabilidad | ¿Apta RAG? | Decisión |
|---|---|---|---|---|---|---|
| `historical_budgets` | ●●●○○ | ●●○○○ | ●●●●○ | ●●●●● | **no** (consistency) | include |
| `meeting_transcripts` | ●●●●● | ●●○○○ | ●●●●○ | ●●●●○ | **no** (consistency) | review |
| `proposal_templates` | ●●●●● | ●●●●○ | ●●●●○ | ●●●●● | sí | include |
| `signed_contracts` | ●●●●○ | ●●●●○ | ●●●●○ | ●●●●● | sí | include |
| `official_rate_card` | ●●●●● | ●●●●● | ●○○○○ | ●●●●● | **no** (actuality) | exclude |

> Las dimensiones no se promedian: cada una es condición necesaria. Una fuente con completitud 5 y fiabilidad 1 no es un 3 — es una fuente cuyos datos están completos y pueden ser falsos, que es lo peor para RAG.

## Frescura declarada vs observada

| Fuente | Declarada | Última actualización | Desfase (días) | Veredicto |
|---|---|---|---:|---|
| `historical_budgets` | monthly | 2026-09-01 | 1 | al día |
| `meeting_transcripts` | weekly | 2026-09-01 | 1 | al día |
| `proposal_templates` | quarterly | 2026-09-01 | 1 | al día |
| `signed_contracts` | monthly | 2026-09-01 | 1 | al día |
| `official_rate_card` | yearly | 2024-01-18 | 958 | 🔴 abandonada (2.6 ciclos sin actualizar) |

## Incluidas pese a no ser aptas tal cual

- **`historical_budgets`** — dimensión débil: **consistency** (2/5). Remediación declarada: NO es apta para RAG tal cual (consistencia 2) y aun así se incluye: la capa tabular de limpieza resuelve exactamente esa dimensión — unifica las 4 grafías de moneda, los 3 formatos de fecha y los 3 tipos de importe— antes de que Pandera valide nada. La puntuación describe la fuente, no la salida del pipeline. Dos budget_id aparecen duplicados con importes divergentes (BUDGET-2024-0312, BUDGET-2023-0188); se resuelven quedándose con la firma más reciente, que es una decisión de negocio, no técnica. Observación aparte: el fichero está fresco pero la última firma real es de 2026-02-24, así que el export corre sin que se cierren presupuestos nuevos.

> La puntuación describe la fuente cruda, no la salida del pipeline. Incluir una fuente débil es legítimo cuando hay una remediación declarada para la dimensión débil; sin nota que lo explique, no lo es.

## Fuentes incluidas

- **`historical_budgets`** (json, 28 registros, 0.031 MB) — negocio: ops-lead@digimevo.com, técnico: data-platform@digimevo.com, origen: `erp-finance-module`
- **`proposal_templates`** (docx, 5 registros, 0.182 MB) — negocio: pre-sales@digimevo.com, técnico: it-ops@digimevo.com, origen: `proposal-template-repo`
- **`signed_contracts`** (pdf, 6 registros, 0.055 MB) — negocio: legal@digimevo.com, técnico: legal-ops@digimevo.com, origen: `document-management-system`

## Fuentes en revisión

- **`meeting_transcripts`** — Dos eras de formato: las 7 de 2024 en adelante traen `[hh:mm:ss] Hablante:` en cada línea; las 5 anteriores son prosa corrida sin atribución. El parser detecta ambas y marca las antiguas con `speaker_attribution: unavailable`, así que el problema técnico está resuelto. Lo que queda pendiente es una decisión de producto, y por eso está en revisión y no incluida: una afirmación sobre presupuesto que no se puede atribuir a nadie no puede pesar lo mismo que una que sí, y el sistema todavía no distingue. Incluirlas es cambiar esta línea a `include`; no hace falta tocar código.

## Fuentes excluidas deliberadamente

- **`official_rate_card`** — El caso de libro: oficialmente autoritativa, en la práctica abandonada. Última modificación enero de 2024 frente a un ciclo declarado anual — 2,6 ciclos sin tocar. Incluirla no metería ruido aleatorio (eso sería fácil de detectar): metería respuestas seguras con tarifas que ya no rigen, que es el peor modo de fallo de un RAG. Excluida hasta que finanzas entregue una versión refrescada. Las tarifas vigentes son contexto pequeño y estable, así que su sitio natural es la capa CAG estática del prompt, no el índice vectorial.

> Excluir es una decisión de higiene, no desidia. Una fuente mala no produce ruido aleatorio (sería fácil de detectar): produce respuestas seguras sobre información incorrecta.

## Datos personales

| Fuente | PII | Tipos | Restricción |
|---|---|---|---|
| `historical_budgets` | sí | client_names, client_codes, contact_emails, contact_phones, account_managers | internal-only |
| `meeting_transcripts` | sí | personal_names, client_names, emails, phone_numbers | internal-only |
| `proposal_templates` | sí | client_names, personal_names | internal-only |
| `signed_contracts` | sí | client_names, personal_names, contract_terms | internal-only |
| `official_rate_card` | no | — | — |

## Última ejecución del pipeline

Inicio: 2026-10-04T17:08:27.966064+00:00 · documentos emitidos: **88**

| Fuente | Decisión | Ficheros | Unidades | Tras limpieza | Documentos | Válidos | Cuarentena | Descartados | Entidades anonimizadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `historical_budgets` | include | 28 | 28 | 23 | 23 | 23 | 0 | 3 | 114 |
| `meeting_transcripts` | review | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `proposal_templates` | include | 5 | 50 | 50 | 50 | 0 | 0 | 0 | 129 |
| `signed_contracts` | include | 6 | 15 | 15 | 15 | 0 | 0 | 0 | 155 |
| `official_rate_card` | exclude | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Duplicados divergentes detectados

- `historical_budgets`: BUDGET-2023-0188, BUDGET-2024-0312 — resueltos por fecha de firma más reciente
