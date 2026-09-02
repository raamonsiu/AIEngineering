# CAG_LIMITS.md — dónde deja de ser viable el CAG, con números medidos

Cierra el bucle entre dos sesiones. Los números de latencia y coste salen del
stress test de la sesión 6 ([`evals/stress/results.csv`](evals/stress/results.csv),
225 turnos reales); los de volumen, de ejecutar los parsers sobre el corpus de
este repositorio. Nada aquí está estimado a ojo.

El árbol de decisión está materializado en
[`app/ingest/architecture.py`](app/ingest/architecture.py) y verificado en
[`tests/test_architecture_decision.py`](tests/test_architecture_decision.py).

## Los cuatro techos

| # | Restricción | Medición | Veredicto |
|---|---|---|---|
| 1 | Ventana de contexto | 19.988 tokens / 128.000 = **15,6%** | ✅ cabe, y con holgura |
| 2 | Coste por consulta | **$0,0030** inyectando el corpus entero | ✅ viable |
| 3 | Latencia | **9,5 s** proyectados vs SLA conversacional de 3 s | ❌ falla |
| 4 | Calidad bajo carga | **100%** de hechos obsoletos sobreviven | ❌ falla |

Basta con que falle uno. Fallan dos — y **ninguno de los dos es el que todo el
mundo nombra primero**.

### 1. Volumen (cabe)

Texto extraíble por los parsers del subsistema `ingest/`, sobre las cinco
fuentes del catálogo:

| Fuente | Chars | ≈ tokens |
|---|---:|---:|
| `historical_budgets` | 16.638 | 4.160 |
| `meeting_transcripts` | 28.036 | 7.009 |
| `proposal_templates` | 11.547 | 2.887 |
| `signed_contracts` | 22.693 | 5.673 |
| `official_rate_card` | 1.040 | 260 |
| **Total** | **79.954** | **19.988** |

A 128K de ventana son el 15,6%. Incluso con el umbral prudente del 70% —
llenar el 95% es técnicamente posible y degrada la calidad — sobra sitio.

El matiz que esta tabla no captura: el corpus **crece linealmente con el
negocio**. Cada cliente nuevo aporta una transcripción y, con suerte, un
presupuesto firmado. Doce reuniones han dado 28.036 chars; a este ritmo el
umbral del 70% (89.600 tokens) llega en torno a las **220 reuniones**. No es
un problema de hoy, pero es la única de las cuatro restricciones que empeora
sola.

### 2. Coste (viable)

Inyectar 19.988 tokens a $0,15/M de gpt-4o-mini son **$0,0030 por consulta**.
Para contraste, el turno 20 medido en el stress test costó $0,001227 con 6.704
tokens de entrada — el corpus completo es 3x eso. A mil consultas diarias son
$3/día. El coste no mata este proyecto.

Y conviene decirlo al revés también, porque la literatura promocional de RAG lo
omite: **RAG sería más caro**. Añade embeddings (uno por chunk, una vez por
ingesta), base vectorial, operación del pipeline de indexación, y
re-indexaciones cada vez que cambie el modelo de embeddings o la estrategia de
chunking. En un corpus que cabe en la ventana, RAG no se elige por coste.

### 3. Latencia (falla)

Aquí hay que tener cuidado con los datos propios. El stress test midió 16.122
ms en el turno 20, pero esa cifra **no sirve para extrapolar**: estaba dominada
por una llamada de compresión que fallaba y añadía ~8 s sin comprar nada (ver
[`evals/stress/REPORT.md`](evals/stress/REPORT.md)). Extrapolar desde ahí daría
48 s, que es ruido, no señal.

La relación limpia sale del barrido de adjuntos, que es el único sin ese
confundido — sesiones de un turno, sin compresión:

| Adjunto | tokens_in | latencia |
|---:|---:|---:|
| 0 KB | 3.560 | 6.040 ms |
| 5 KB | 4.284 | 6.334 ms |
| 20 KB | 6.143 | 6.026 ms |
| 50 KB | 9.834 | 7.204 ms |

Ajuste sobre los 36 puntos: `latencia_ms ≈ 0,222 × tokens + 5.078`.

A corpus completo (19.988 tokens): **9,5 s**. El SLA conversacional son 3 s.
Falla por un factor de 3, y el término que domina no es la pendiente sino el
intercepto de 5 s — es decir, **ni siquiera un corpus vacío cumpliría el SLA
con este modelo**. Esto es importante para no engañarse: reducir contexto no
arregla la latencia de este sistema.

### 4. Calidad bajo carga (falla)

Es el techo más subestimado y el único que no se deduce de una fórmula: hace
falta una curva de recall medida. El stress test la produjo.

- **Hechos obsoletos: 6/6 = 100% sobreviven.** React sigue en
  `mentioned_technologies` 15 turnos después de cancelarse; el presupuesto de
  30.000 € sigue presente 12 turnos después de sustituirse por 80.000 €. La
  memoria acumula y no retracta.
- **Recall de contenido de adjunto: 0,185 (5 KB) → 0,037 (20 KB) → 0,000
  (50 KB).** Hay un rango en el que el documento entra en el prompt, se paga, y
  el modelo no lo usa. Eso es *lost in the middle* medido en este sistema.

El nombre del proyecto, en cambio, nunca se pierde (1,000 en 180 turnos). El
CAG de este repositorio no tiene un problema de olvido; tiene uno de
**obsolescencia**.

## La decisión

```
CAGViability(
    fits_in_context_window   = True,   # 15,6% de la ventana
    cost_per_query_acceptable = True,  # $0,0030
    latency_acceptable       = False,  # 9,5 s vs 3 s
    quality_holds_with_load  = False,  # 100% de obsoletos sobreviven
)
# is_viable() -> False
# recommend_architecture(...) -> Architecture.PURE_RAG
```

Pero el árbol no llega ahí por los números. Llega antes, por los dos ejes
funcionales, y conviene decirlo en ese orden ante un stakeholder:

**Trazabilidad.** Si el sistema propone 80.000 € y comercial no puede decir qué
precedentes lo justifican, la estimación es indefendible ante el cliente. CAG
inyecta el corpus entero y no puede atribuir una afirmación a un fragmento
concreto. Esto no se negocia por tamaño: un corpus de 500 tokens que necesite
citar sigue necesitando RAG.

**Control de acceso.** Las transcripciones y los contratos son confidenciales
por cliente y por equipo. Todo lo que entra en el prompt es visible para quien
preguntó. Un CAG no tiene dónde poner un permiso.

Los cuatro techos cuantitativos confirman la decisión; los dos ejes funcionales
la toman. Es la diferencia entre *"no cabe"* y *"no sirve"*.

## Qué se queda en CAG

RAG **además de** CAG, no en lugar de. Lo que es pequeño, estable y no
necesita citarse sigue siendo más simple, más barato y más predecible como
contexto estático:

- El glosario de tecnologías y la terminología de la empresa.
- La estructura estándar de presupuesto (fases, formato de salida) — ya está
  en `app/prompts/` y en `ESTIMATION_EXAMPLES`.
- Las tarifas oficiales vigentes. Nótese que el `official_rate_card` del
  catálogo está **excluido del RAG** por obsoleto; su sustituto, cuando
  finanzas lo entregue, es contexto estático, no un documento a vectorizar: es
  una tabla de 260 tokens que cambia una vez al año.

Lo construido en el módulo 2 no se tira. Cambia de papel.

## Límites de este análisis

- **La proyección de latencia es una extrapolación**, de 9.834 tokens medidos a
  19.988. El ajuste es lineal sobre un rango que no llega a cubrir el corpus
  entero, así que la cifra de 9,5 s es un orden de magnitud, no una promesa.
- **No se ha medido *lost in the middle* sobre el corpus completo**, solo sobre
  adjuntos de hasta 50 KB. La caída a recall 0,000 a 50 KB es la evidencia más
  cercana, y apunta en la dirección correcta, pero no es el mismo experimento.
- **Un solo modelo** (gpt-4o-mini). El intercepto de 5 s es suyo; otro modelo
  movería los cuatro techos y habría que volver a medir.
