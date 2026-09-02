# Stress test del CAG: dónde rompe

**Baseline cuantitativo previo a RAG.** 225 turnos reales contra
`POST /api/v1/sessions/{id}/estimate`, 0 errores, 46,8 min, 0,2366 USD.
Datos crudos en [`results.csv`](results.csv); todas las tablas de aquí se
regeneran con `uv run python -m evals.stress.analyse`.

## Montaje

| | |
|---|---|
| Modelo | `gpt-4o-mini-2024-07-18` (único proveedor, sin fallback activado) |
| Constantes | `MAX_TURNS=6`, `MAX_ATTACHMENT_WORDS=8000`, `ANCHOR_DETECTION_MODE=heuristic` — **sin tocar** |
| Barrido de turnos | 3 escenarios × 3 repeticiones × 20 turnos = 180 filas, sin adjunto |
| Barrido de adjuntos | 3 escenarios × 5 tamaños × 3 repeticiones × 1 turno = 45 filas |
| Presupuestos | `LatencyBudgetMetric(10000)`, `CostBudgetMetric(0.001)` |

Dos barridos en vez del producto cartesiano que esboza el enunciado: una
conversación de 20 turnos que además arrastra un PDF de 50 KB en cada turno
no permite saber si el turno 14 fue lento por el historial o por el adjunto.
Cada barrido varía una sola cosa. La escalera N ∈ {1,3,6,10,20} se lee de
las filas del barrido largo, porque una sesión de 20 turnos **contiene** sus
propios prefijos de 1, 3, 6 y 10.

## Tabla resumen

| barrido | escenario | adj. KB | n | P50 lat (ms) | P95 lat (ms) | coste total (USD) | hit exacta | hit semántica | recall medio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| turns | growing | 0 | 60 | 14709 | 18220 | 0,0644 | 0% | 0% | 0,844 |
| turns | pivot | 0 | 60 | 13526 | 18598 | 0,0581 | 0% | 0% | 1,000 |
| turns | contradiction | 0 | 60 | 14642 | 19345 | 0,0679 | 0% | 0% | 1,000 |
| attachments | growing | 0 | 3 | 5927 | 6649 | 0,0021 | 0% | 0% | 1,000 |
| attachments | growing | 5 | 3 | 6248 | 7175 | 0,0024 | 0% | 0% | 1,000 |
| attachments | growing | 20 | 3 | 5354 | 5747 | 0,0032 | 0% | 0% | 1,000 |
| attachments | growing | 50 | 3 | 5946 | 6042 | 0,0049 | 0% | 0% | 1,000 |
| attachments | growing | 100 | 3 | 5616 | 7365 | 0,0021 | 0% | 0% | 1,000 |
| attachments | pivot | 50 | 3 | 6347 | 12168 | 0,0049 | 0% | 0% | 1,000 |
| attachments | contradiction | 50 | 3 | 7813 | 8290 | 0,0054 | 0% | 0% | 1,000 |

*(filas de `pivot`/`contradiction` a 0/5/20/100 KB omitidas por brevedad; están en el CSV)*

**El hit rate 0% no es casualidad ni fallo de instrumentación.**
`estimate_in_session` salta ambas caches a propósito: la clave de la exacta es
la descripción, y en una sesión el mismo texto significa cosas distintas según
el historial. El campo `cache_hit_kind` se emite igualmente para que ese 0% sea
*visible* en el dataset en vez de una ausencia que hay que saberse de memoria.

## Curva 1 — latencia vs tokens_in

Separada por barrido, porque agregada **miente**:

| tokens_in | barrido turns: n / P50 (ms) | barrido attachments: n / P50 (ms) |
|---|---:|---:|
| 0–4.000 | 16 / 6.759 | 19 / 5.616 |
| 4.000–5.500 | 47 / 7.652 | 8 / 7.788 |
| 5.500–7.000 | 102 / **15.141** | 9 / **5.747** |
| 7.000+ | 15 / 16.443 | 9 / 6.347 |

A **igualdad de tokens** (5.500–7.000), el barrido conversacional tarda 2,6x
más que el de adjuntos. La latencia no es función del tamaño del contexto.

## Curva 2 — coste acumulado vs turno

| turno | contradiction | growing | pivot | coste medio/turno | latencia media (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,00089 | 0,00069 | 0,00066 | 0,000747 | 7.882 |
| 3 | 0,00247 | 0,00228 | 0,00218 | 0,000811 | 7.061 |
| 6 | 0,00542 | 0,00517 | 0,00480 | 0,001027 | 7.429 |
| 10 | 0,00991 | 0,00963 | 0,00890 | 0,001121 | 15.312 |
| 20 | 0,02263 | 0,02146 | 0,01936 | 0,001227 | 16.122 |

El turno 20 cuesta **1,64x** el turno 1 (0,001227 vs 0,000747 USD). Se aplana
porque la ventana deslizante le pone techo a `tokens_in` por diseño.

## Curva 3 — recall vs N

| turno | contradiction | growing | pivot | recall `project_name` | hechos obsoletos aún presentes |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,000 | 1,000 | 1,000 | 1,000 | 3/3 |
| 3 | 1,000 | 1,000 | 1,000 | 1,000 | 6/6 |
| 6 | 1,000 | 0,778 | 1,000 | 1,000 | 6/6 |
| 10 | 1,000 | 0,809 | 1,000 | 1,000 | 6/6 |
| 20 | 1,000 | 0,833 | 1,000 | 1,000 | **6/6** |

El nombre del proyecto **nunca** se pierde: 1,000 en los 180 turnos. Pero los
hechos obsoletos sobreviven el **100%** de las veces — React sigue en
`mentioned_technologies` 15 turnos después de cancelarse, y el presupuesto de
30.000 € sigue presente 12 turnos después de sustituirse por 80.000 €.

## Barrido de adjuntos

| adj. KB | chars extraídos | aceptados | P50 lat (ms) | tokens_in medios | coste medio | recall del adjunto |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9/9 | 5.767 | 3.560 | 0,000716 | n/a |
| 5 | 5.136 | 9/9 | 6.248 | 4.284 | 0,000854 | 0,185 |
| 20 | 20.754 | 9/9 | 5.747 | 6.143 | 0,001124 | 0,037 |
| 50 | 51.569 | 9/9 | 6.347 | 9.834 | 0,001688 | **0,000** |
| 100 | **205** | **0/9** | 6.679 | 3.685 | 0,000758 | rechazado |

A 100 KB (10.284 palabras > `MAX_ATTACHMENT_WORDS=8000`) el adjunto **no se
trunca: se descarta entero**. Los 205 chars son la nota de fallo, y tanto
tokens como coste vuelven al nivel del baseline sin adjunto. La estimación se
devuelve con HTTP 200.

## Cruces de SLA de latencia

| SLA | turnos que pasan | primer turno donde el P50 lo incumple |
|---:|---:|---:|
| 4.000 ms | 1/180 (1%) | 1 |
| 8.000 ms | 42/180 (23%) | **7** |
| 10.000 ms | 55/180 (31%) | **7** |
| 15.000 ms | 109/180 (61%) | **7** |

## Lectura

**¿A partir de qué turno empieza a romperse, y qué dimensión domina?** El codo
está en el **turno 7**, y es un escalón, no una pendiente: la latencia media
pasa de 7.429 ms (turno 6) a 16.682 ms (turno 7), un **2,25x en un solo
turno**, mientras `tokens_in` solo sube un 5% (5.434 → 5.711). Todo SLA de 8 s
en adelante se rompe exactamente ahí. El turno 7 es el primero en que
`len(messages) > MAX_TURNS*2` y la política de compresión se dispara — y la
dimensión que domina **no es el contexto, es una llamada lateral que falla**.
El summarizer invoca `gpt-5-nano` con `max_tokens=1000`; al ser modelo de
razonamiento agota ese presupuesto razonando y devuelve
`IncompleteOutputException`. El `except` falla abierto, devuelve el summary
previo y la petición responde 200. Resultado: `summary_chars = 0` en **180 de
180 turnos**, los turnos desalojados se pierden sin rastro, y cada turno a
partir del 7 paga ~8 s de latencia por una llamada que no compra nada. La
prueba está en la curva 1: a igualdad de tokens (5.500–7.000), los turnos
conversacionales tardan 15.141 ms y los de adjunto 5.747 ms. El coste, en
cambio, **no es el problema**: 1,64x del turno 1 al 20, y 0,021 USD por
conversación completa de 20 turnos. Esto es degradación silenciosa de manual —
ningún test en rojo, ninguna excepción al cliente, y la mitad de la
arquitectura de memoria no existe.

**¿Qué constituiría el caso límite que justifica saltar a RAG?** Tres, y el
tercero es el que de verdad decide. Primero, el adjunto: el recall de su
contenido cae de 0,185 (5 KB) a 0,037 (20 KB) a **0,000 (50 KB)**, y a 100 KB
se descarta entero con HTTP 200. Hay un rango de tamaños — entre 5 y 50 KB —
en el que el documento sí entra en el prompt, sí se paga (el coste sube 2,4x) y
el modelo aun así **no lo usa**: el peor cuadrante posible, pagar por contexto
que no rinde. Segundo, el coste por turno crece mientras el valor no: a partir
del turno 7 se reenvía el mismo historial en cada llamada para obtener una
estimación que cambia poco. Tercero, y decisivo: la memoria **acumula pero no
retracta**. Los hechos obsoletos sobreviven el 100% de las veces en los turnos
medidos, porque `ProjectMetadata.merge_with` hace unión de listas y solo
sobrescribe escalares con valores no nulos — React sigue presente 15 turnos
después de cancelarse. Eso es una decisión de diseño correcta contra el olvido
del extractor, pero convierte la memoria en monótona creciente: un CAG puede
recordar más, nunca puede *corregirse*. El caso límite que obliga a RAG no es
"el contexto está lleno", es **cualquier conversación en la que un hecho se
sustituya por otro**, porque ningún tamaño de ventana arregla una memoria que
no sabe borrar. Con `project_name` al 1,000 en 180 turnos, el CAG no tiene un
problema de olvido; tiene un problema de obsolescencia.

## Límites conocidos de esta medición

- **El coste está infravalorado a partir del turno 7.** La llamada fallida del
  summarizer consume tokens, pero como lanza excepción nunca llega a
  `record_llm_call`; por eso `llm_calls` vale 2,00 en todos los turnos y nunca 3.
  La latencia sí la recoge, porque se mide con reloj de pared alrededor del turno.
- **`MemoryDriftMetric` es substring case-insensitive**, no semántica. Los
  `aliases` cubren las formas de superficie plausibles, pero un parafraseo
  genuino cuenta como olvido. Determinismo elegido sobre sofisticación: un
  LLM-as-judge aportaría a la medición la misma no determinación que la
  medición existe para cuantificar.
- **`recall` mide anclas + metadata, no el triplete completo.** Con el summary
  vacío en el 100% de los turnos, la curva 3 describe un CAG al que le falta una
  de sus tres ranuras de memoria. Las anclas sí funcionaron donde debían:
  `contradiction` promovió 4 mensajes (los dos turnos de presupuesto); `growing`
  y `pivot`, 0.
- **Un único proveedor y un único modelo**, por diseño, para que las curvas sean
  comparables. Nada aquí se traslada a otro modelo sin volver a medir.
