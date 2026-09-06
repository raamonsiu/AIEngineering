# Sanity check de embeddings — tres parejas

`text-embedding-3-small`, 1536 dimensiones, medido el 2026-10-04 con
[`scripts/compare.py`](../../scripts/compare.py). No es una validación formal
del modelo: es la comprobación mínima de que el pipeline funciona de punta a
punta y de que los vectores separan lo cercano de lo lejano.

| Pareja | Textos | Esperado | Obtenido | |
|---|---|---|---|---|
| **A** — cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" ↔ "Authorization service using JSON Web Tokens for a banking application" | > 0,6 | **0,5957** | ✗ por 0,004 |
| **B** — no relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" ↔ "Database migration from MySQL to PostgreSQL with zero downtime" | < 0,4 | **0,1920** | ✓ holgado |
| **C** — genéricos | "Backend services" ↔ "API development" | sin expectativa | **0,5407** | — |

## Comentario

El orden sale bien (A > C > B) pero los valores absolutos no significan lo que
parece. La pareja A se queda en 0,5957, justo por debajo del 0,6 orientativo,
pese a describir exactamente el mismo sistema: no comparten ni una palabra
relevante ("JWT" contra "JSON Web Tokens", "fintech" contra "banking") y el
modelo todavía paga bastante por la superficie léxica. Lo llamativo es la
pareja C: dos frases genéricas, sin un solo término en común, puntúan 0,5407,
a 5 centésimas de un acierto real. Es decir, la distancia entre "lo mismo
dicho de otra forma" y "las dos cosas son vagas" cabe en el margen de ruido,
mientras que el suelo de lo no relacionado (0,1920) está clarísimo.

La consecuencia práctica para la sesión 08 es directa: un umbral absoluto no
vale. Cortar en 0,6 habría descartado la pareja A, que es un acierto; cortar
en 0,5 habría aceptado la C, que no tiene contenido. El coseno de este modelo
sirve para **ordenar**, no para decidir, así que el retrieval debe ser top-k y
el filtrado duro tiene que venir de los metadatos del chunk, no de la
similitud.

## Reproducir

```bash
uv run python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" \
  --text-b "Authorization service using JSON Web Tokens for a banking application"
```

Añade `--verbose` para ver en stderr el log del embedder (tokens y latencia).
Dentro del contenedor, lo mismo con `docker compose exec cag-estimator`.

## El pipeline completo, de paso

`POST /api/v1/embeddings/ingest` con [`data/budgets_sample.json`](../../data/budgets_sample.json)
(15 presupuestos, 4 sectores, 10 tecnologías principales):

| | |
|---|---|
| Chunks emitidos | 67 (uno por componente) |
| Tokens | 6.483 — entre 87 y 114 por chunk |
| Llamadas a la API | 1 (lote de 100; 67 entran de sobra) |
| Latencia del lote | 2.401 ms |
| Coste estimado | 0,00012966 USD |
| Respuesta | 2,0 MB — 1536 floats por chunk pesan ~30 KB en JSON |

Dos cosas que confirma esta ejecución. La primera: `tokens_billed=6483` y
`tokens_estimated=6483`, o sea que el contador local de tiktoken coincide
exacto con el que factura OpenAI, así que el coste se puede calcular sin
llamar a nadie. La segunda: ningún chunk se acerca al límite de 8.191 tokens
del modelo — el máximo es 114. Partir componentes por tamaño no habría tenido
nada que partir, que es justo el argumento para no hacerlo.
