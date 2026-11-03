# SABER: Propuesta de Proyecto (v2)

**Ejecución en español verificada por comportamiento para robots VLA**
*SABER = Spanish-Aware, Behavior-verified Execution for Robots*

Ian Carlos Hernández Robles · FrED Factory · Tecnológico de Monterrey
Versión 2 · 3 de noviembre de 2026 (sustituye a la v1 del 28 de septiembre de 2026)

---

## 0. Cambios respecto a la v1

| # | Cambio | Motivo | Consecuencia |
|---|---|---|---|
| 1 | El brazo pasa de un SO-101 a un **uFactory xArm 6** del laboratorio | Decisión del equipo (24 oct 2026) | Desaparece el kit de USD 400–750; cambian teleoperación, integración con LeRobot, seguridad y presupuesto |
| 2 | Se retira "bajo costo" del título y de C1 | Un xArm 6 es un cobot industrial, no hardware de aficionado | El argumento de accesibilidad se traslada al **modelo y al cómputo** (SmolVLA en una laptop), no al brazo |
| 3 | Fase A con **3 objetos** (desarmador, pinzas, llave) en lugar de 6 | La v1 pedía 150 demos (50 por objetivo) con 6 objetivos: no cuadraba | Los otros 3 objetos pasan a Fase B |
| 4 | Licencia del código: **MIT** | El repositorio ya tiene licencia MIT; la v1 decía Apache-2.0 | Se corrige la inconsistencia |
| 5 | Cronograma rebaselinado al 3 de noviembre | Al 3 de noviembre el repositorio solo contiene la licencia: el plan de 10 semanas iba en la semana 6 sin código | Las compuertas G2–G3 se recorren; G3 probablemente cae en la Fase B |
| 6 | Registro en Expoingenierías | Requisito académico (materia MR-2035) | Se agrega una demo de expo como entregable intermedio |

---

## 1. Introducción

### 1.1 Contexto y decisión de alcance

El AGV para tarimas no es el anfitrión adecuado para un VLA: su navegación es una pila clásica y madura (SLAM, lidar de seguridad), y los camiones industriales sin conductor tienen su propia norma de seguridad (ISO 3691-4; verificar antes de citar). SABER lleva el VLA a la manipulación de mesa, donde la evidencia publicada sí lo respalda. Incluso el trabajo de VLA con movilidad mantiene clásica la navegación: AnywhereVLA [1] combina SLAM con una cabeza SmolVLA y reporta 46 % de éxito global.

**Qué hace el robot.** Un xArm 6 surte herramientas etiquetadas (desarmador, pinzas, llave en la Fase A) desde la mesa hacia una charola de kit en un laboratorio de enseñanza. Complementa el proyecto del AGV: el AGV mueve la caja y el brazo hace el último metro de recolección.

**Por qué español.** Los operadores de un laboratorio mexicano hablan español. Los VLA abiertos se entrenan sobre todo con instrucciones en inglés, y los estudios multilingües existentes son en simulación [2, 3], salvo una demostración en robot real [4].

**Por qué un xArm 6 y qué se pierde.** El xArm 6 (6 grados de libertad, carga útil de 5 kg y alcance de unos 700 mm según el fabricante; verificar con la ficha del equipo del laboratorio) ofrece repetibilidad y seguridad de cobot que el SO-101 no tiene, y ya está disponible. A cambio:
- SmolVLA se preentrenó sobre todo con datos de brazos SO-100; el xArm 6 es otra morfología. Se espera necesitar más demostraciones.
- No hay brazo líder: la teleoperación requiere otro dispositivo (sección 2.2.1).
- LeRobot no trae, hasta donde se sabe, un driver nativo para el xArm 6 real: hay que escribirlo con el xArm-Python-SDK (verificar en la versión vigente de LeRobot).
- El argumento de "hardware de bajo costo" ya no aplica al brazo. Se sostiene para el modelo (450 M parámetros) y el cómputo (laptop con RTX 5060).

### 1.2 Planteamiento del problema

Un VLA ajustado con órdenes en inglés no falla de forma ruidosa cuando le habla una persona en español: se mueve hacia algo, con seguridad, y con frecuencia hacia lo equivocado. En LIBERO-PRO los modelos siguen ejecutando el comportamiento entrenado aun con instrucciones corruptas o sin sentido [5]. LIBERO-Plus encuentra que repiten la acción original aunque la instrucción cambie [6].

En LIBERO-Goal, donde la escena es fija y la instrucción es la única señal de la tarea, el éxito con español cae de 97.9 % a 11.2 % en OpenVLA-OFT y de 97.6 % a 14.2 % en π0.5 [2] (simulación). Sin ninguna instrucción, esos modelos logran 10.0 % y 9.0 %, casi igual que con francés (9.5 %) [3]. En esa suite, una instrucción no inglesa equivale a no dar instrucción.

**Ejecución errónea silenciosa:** episodio ejecutado en el que el brazo agarra o se mueve hacia un objeto distinto del ordenado, sin abstención, detención ni alarma antes del contacto.

**Preguntas de investigación**

- **RQ1. Diagnóstico.** ¿Con qué frecuencia un VLA abierto y pequeño ejecuta mal órdenes en español, Spanglish y voz ruidosa en un robot real, respecto de controles sin instrucción y de azar?
- **RQ2. Arreglos baratos.** ¿Cuánto de la brecha recuperan los arreglos sin tiempo de robot (canonicalización con un LLM multilingüe, reetiquetado de demostraciones con paráfrasis en español) y qué riesgo residual queda?
- **RQ3. Compuerta.** ¿Marginalizar sobre interpretaciones candidatas detecta el riesgo residual antes del movimiento mejor que las señales de solo texto, de distribución de embeddings y de la propia política, y con qué costo de latencia?
- **RQ4. Frontera y costo.** En un vocabulario de laboratorio fijo, ¿en qué tipos de orden un VLA supera a una cascada de analizador más ACT, medido en éxito, latencia y horas de robot?

---

## 2. Desarrollo

### 2.1 Hipótesis y escalera de afirmaciones

C1 y C2 bastan para un artículo publicable; C3 lo acerca a una pista principal. La confianza en la novedad es juicio del autor, no una medición.

| # | Afirmación | Trabajo previo más cercano | Confianza en novedad | Si falla |
|---|---|---|---|---|
| C1 | La ejecución errónea silenciosa con español, Spanglish o ASR es común en un robot real con un VLA abierto y pequeño, y los controles ciegos al lenguaje lo demuestran | [2, 3] (simulación); [4] (una demostración real) | Media-alta | Estudio diagnóstico para IEEE CASE o RA-L |
| C2 | Banco abierto: léxico mexicano de laboratorio, demostraciones en formato LeRobot para xArm 6 y banco de puntos de decisión fuera de línea | [5, 6] (simulación, inglés) | Media | Aún publicable; más débil |
| C3 | IMG reduce el riesgo de objeto equivocado con igual cobertura frente a compuertas de texto, de distribución y de varianza de acción | [7, 8, 9, 10] | Baja-media | Se retira como titular; queda como análisis de señales |
| C4 | Mapa de costos: dónde basta un analizador más ACT y dónde hace falta un VLA, en latencia, cómputo y horas de robot | [11, 12] | Baja-media | Resultado secundario |

**Hipótesis preregistradas** (los umbrales se congelan en G0):

- **H1.** Con ajuste solo en inglés, la precisión de objetivo con español crudo, Spanglish y voz queda a menos de 15 puntos del control sin instrucción y significativamente por debajo del inglés.
- **H2.** La canonicalización con la mejor traducción recupera al menos 80 % del éxito en inglés con español formal, pero menos con Spanglish y voz ruidosa, con una tasa de objeto equivocado residual de al menos 10 %.
- **H3.** IMG supera a la mejor compuerta base en AUROC por al menos 0.03, con intervalo bootstrap al 95 % que excluye el cero, en al menos dos de tres tipos de cambio.
- **H4.** Con 80 % de cobertura, IMG reduce al menos a la mitad la tasa de objeto equivocado entre decisiones ejecutadas respecto de la canonicalización sola.
- **H5 (exploratoria).** Un analizador más ACT iguala a SmolVLA en órdenes dentro del vocabulario con latencia 10 veces menor o mejor, y falla con objetos nuevos.

Si H1 falla se cae la premisa; si H2 falla desaparece la necesidad de una compuerta; si H3 o H4 fallan se retira C3. Todo resultado se reporta.

### 2.2 Enfoque técnico

Tubería: **escuchar → transcribir → interpretar → verificar (IMG) → actuar**. Solo *verificar* es nuevo.

1. **Escuchar:** voz o texto en ES, ES-MX, Spanglish o EN.
2. **Transcribir:** Whisper [16] con varias decodificaciones conservadas como hipótesis (n-best).
3. **Interpretar:** un LLM/VLM multilingüe pequeño ve el fotograma y propone K instrucciones canónicas en inglés.
4. **Verificar (IMG):** SmolVLA corre por candidata; los movimientos resultantes se agrupan. Si el consenso ≥ τ, actúa; si no, pregunta en español y espera.
5. **Actuar:** trozo de acción de consenso bajo límites duros (sección 2.2.5).

#### 2.2.1 Hardware y modelos

| Componente | Elección | Notas |
|---|---|---|
| Brazo | uFactory xArm 6 del laboratorio | Confirmar disponibilidad, horas de uso, versión de firmware y controlador |
| Pinza | xArm Gripper (confirmar modelo) | Si no existe, es la mayor partida del presupuesto |
| Cámaras | 2 USB: muñeca y frontal | Montaje de muñeca impreso en la Bambu Lab A1 |
| Teleoperación | **Opción preferida: GELLO para xArm** (réplica cinemática con servos Dynamixel). Alternas: SpaceMouse en espacio cartesiano; modo manual (guiado a mano) del xArm | El guiado a mano mete la mano del operador en la imagen y contamina las demos; usarlo solo como último recurso |
| Integración | Clase `Robot` propia para LeRobot sobre el xArm-Python-SDK | Verificar la guía de integración de hardware de la versión de LeRobot fijada |
| Política | SmolVLA [15] (450 M) ajustada por tarea | Espacio de acción: 6 articulaciones + pinza (7), distinto del SO-100 preentrenado |
| ASR | Whisper en la laptop | n-best por beam search o temperatura |
| Intérprete | LLM/VLM multilingüe abierto y pequeño | Se elige con un piloto de 200 enunciados |
| Cómputo | Laptop con RTX 5060 (inferencia); GPU rentada (ajuste) | |

El modelo de lenguaje de SmolVLA es SmolLM2-360M, cuya ficha indica que entiende y genera principalmente inglés [17]. Por eso se espera una brecha en español, pero no se da por supuesta.

#### 2.2.2 Compuerta de Interpretación Marginalizada (IMG)

El enunciado *u* y el fotograma *o* se tratan como evidencia de una instrucción canónica latente *c* (en inglés, dentro del vocabulario de ajuste). El intérprete muestrea K candidatas c₁…c_K con pesos w_k (probabilidades de secuencia renormalizadas; uniformes como ablación). Por candidata, el prefijo del VLA se calcula una vez y se muestrean S trozos de acción del experto de flujo. Costo: K pasadas de prefijo + K·S pasadas del experto.

Cada trozo se convierte en una característica de decisión φ: posición del efector final por **cinemática directa del xArm 6** (URDF oficial de uFactory, no parámetros DH inventados) en el horizonte H, más la orden de la pinza. Los trozos se agrupan con un radio relativo a la dispersión dentro de cada candidata. Con masas de grupo m_j = Σ_k w_k · (fracción de muestras de c_k en el grupo j):

- m* = max_j m_j
- H_B = −Σ_j m_j log m_j
- R = Var_k[E[φ | c_k]] / E_k[Var[φ | c_k]]

La compuerta ejecuta el grupo modal si m* ≥ τ; si no, se abstiene y pregunta en español nombrando los dos objetivos más probables. H_B es el análogo de la entropía semántica [19] sobre comportamientos. Bajo aproximación gaussiana con igual covarianza dentro de candidatas, la información mutua entre índice de candidata y trozo está acotada por ½ log(1 + R); se usa como motivación, no como garantía.

#### 2.2.3 Dónde corre y cómo se calibra

La compuerta corre al inicio del episodio y en cada replanteo hasta que la pinza cierra; después la política corre con una sola vista. El lenguaje importa sobre todo en pocos pasos de decisión [2], por lo que una compuerta previa al movimiento es la colocación natural. τ se fija con puntos de decisión fuera de línea etiquetados, con control de riesgo de estilo conformal dividido [20], acotando al nivel α la tasa de objeto equivocado entre decisiones ejecutadas. La garantía es marginal y vale solo si el despliegue se parece a la calibración; la calibración mezcla tipos de cambio y se reportan violaciones por condición.

#### 2.2.4 Compuertas base

- **G-text:** acuerdo solo de texto entre las K candidatas (coseno de embeddings, sin VLA).
- **G-conf:** log-probabilidad media del ASR y probabilidad de secuencia del intérprete.
- **G-ood:** distancia del embedding de la instrucción al conjunto de ajuste.
- **G-act:** entropía de agrupamiento con una sola vista sobre muestras de flujo (análogo a ACE [9] y a una versión ligera de UPS [7]); equivale a IMG con K = 1.
- **G-int:** Mahalanobis del estado oculto + consistencia de trozos, al estilo de VLA-FAIL [21] (según su resumen).
- **G-pert:** desacuerdo por perturbación de pesos de bajo rango, al estilo de PFD [22] (Fase B).

#### 2.2.5 Supervisor de seguridad y movilidad opcional

El xArm 6 tiene bastante más fuerza que un SO-101: el supervisor ya no es opcional.

- **En el controlador del xArm** (funciones del xArm-Python-SDK; verificar nombres en la versión instalada): sensibilidad de colisión alta, modo reducido con velocidad TCP máxima baja, límite cartesiano del TCP (caja de trabajo) y límites articulares.
- **En software, entre política y SDK:** límites articulares, tope de velocidad y aceleración por paso, caja de trabajo, rechazo de acciones con saltos mayores a un umbral, y paro si se pierde la cámara o la política tarda más de un plazo.
- **Paro de emergencia físico** al alcance del operador en todo ensayo.

El supervisor es independiente del modelo y **no está certificado** como dispositivo de seguridad. El prototipo no pretende cumplir ISO 10218 ni ISO/TS 15066 y no debe describirse como certificado.

Movilidad: la demo con base tipo LeKiwi de la v1 queda **suspendida**: la base no carga un xArm 6. La integración con el AGV queda como decisión del PM.

### 2.3 Diseño experimental

Dos flujos de evidencia: uno grande, fuera de línea y en simulación (potencia), y uno pequeño en robot real (validez).

#### 2.3.1 Tareas, objetos y datos

- **T1:** "toma <objeto> y ponlo en la charola", con los tres objetos visibles y posiciones aleatorizadas en cinco zonas, de modo que la instrucción sea la única pista del objetivo.
- **T2 (Fase B):** "pon <objeto> en el contenedor izquierdo o derecho" (lenguaje espacial).
- **Objetos de la Fase A:** desarmador (destornillador, *screwdriver*), pinzas (*pliers*), llave (*wrench*). Para A1 se agrega unas pinzas de precisión como prop de ambigüedad. Flexómetro, cinta y carrete de estaño pasan a la Fase B.
- **Datos:** 50 a 75 demostraciones teleoperadas por objetivo (150 a 225 en total) en las cinco zonas, con instrucciones de entrenamiento **solo en inglés**. Un 20 % adicional en zonas reservadas para evaluación. Estimación: 12 a 20 horas de operador, más que en la v1 por el cambio de morfología y la teleoperación sin brazo líder.

#### 2.3.2 Condiciones de lenguaje

30 enunciados por condición, redactados con dos hablantes nativos mexicanos, ampliados con un LLM y revisados por humanos.

| Código | Condición | Ejemplo (objetivo = desarmador) | Correcto |
|---|---|---|---|
| E0 | Inglés canónico | "Pick up the screwdriver and put it in the tray" | Ejecutar |
| E1 | Paráfrasis en inglés | "Grab the screwdriver and drop it in the tray" | Ejecutar |
| S1 | Español formal | "Toma el destornillador y ponlo en la charola" | Ejecutar |
| S2 | Español mexicano coloquial | "Pásame el desarmador y ponlo en la charola" | Ejecutar |
| S3 | Spanglish | "Agarra el screwdriver y ponlo en la tray" | Ejecutar |
| V1 | Voz, cuarto silencioso | S2 hablado y transcrito | Ejecutar |
| V2 | Voz, ruido de laboratorio (dos SNR) | S2 sobre ruido de máquinas grabado | Ejecutar o abstenerse |
| A1 | Referente ambiguo | "Pásame las pinzas" con dos tipos visibles | Abstenerse y preguntar |
| O1 | Objeto ausente | "Pásame el multímetro" sin multímetro | Abstenerse |

#### 2.3.3 Sistemas y ensayos reales

40 ensayos pareados por celda salvo indicación.

| ID | Sistema | Condiciones | Ensayos |
|---|---|---|---|
| S0 | SmolVLA-EN con enunciado crudo + controles sin instrucción y barajado | E0, S2, S3, V2 | 240 |
| S1 | S0 + canonicalización (translate-test) | E0, S2, S3, V1, V2; A1 y O1 con n = 20 | 240 |
| S2 | S1 + IMG y arbitraje por grupo modal | S2, S3, V2; A1 y O1 con n = 20 | 160 |
| S3 | SmolVLA reetiquetado en español y Spanglish, enunciado crudo | S2, S3 | 80 |
| S4 | Analizador + ACT por objeto | E0, S2 | 80 |
| | **Total** | | **800** |

Unas 20 a 27 horas de robot a 1.5–2 minutos por ensayo (estimación; el reseteo de escena en el xArm se mide en el piloto). Si el tiempo de robot escasea, la prioridad es S0 > S1 > S2 > S4 > S3.

#### 2.3.4 Métricas

SR, WOR, FCR, AR, riesgo (WOR entre ejecutados), cobertura (1 − AR), AURC [24], latencia por decisión y horas de robot. Dos anotadores etiquetan desde video, ciegos a la condición cuando sea posible; se reporta kappa de Cohen. Definiciones en el Apéndice C.

#### 2.3.5 Banco de puntos de decisión fuera de línea (ODPB)

Para cada demostración reservada se consulta la política en t = 0 y en los replanteos previos al agarre, bajo cada condición de lenguaje. Una decisión es correcta si el efector final al final del primer trozo (por cinemática directa del xArm 6) queda más cerca del objetivo verdadero que de los otros dos. Esto da miles de puntos etiquetados sin tiempo de robot. Los ensayos reales verifican después que el AUROC fuera de línea ordene las compuertas igual que el lazo cerrado.

#### 2.3.6 Complemento en simulación

LIBERO-Goal [25] traducido al español y a tres idiomas más. Se corren S0, S1 y S2 con 10 tareas × 20 episodios por condición, con un punto de control público de SmolVLA para LIBERO si existe (verificar) o un ajuste propio. Una RTX 4090 se renta por cerca de 0.34–0.69 USD/h (RunPod, junio de 2026; verificar).

#### 2.3.7 Estadística, ablaciones y amenazas a la validez

**Estadística.** 40 ensayos pareados por celda detectan aproximadamente 30 puntos con 80 % de potencia (bilateral, α = 0.05; cálculo propio, depende de la tasa de discordancia; verificar con simulación antes de G0). La Fase B sube celdas clave a n ≥ 100. Wilson para proporciones, McNemar para pares, bootstrap por conglomerados de escena para AUROC y AURC, Holm sobre comparaciones preregistradas, y parada secuencial de STEP [26] si escasean ensayos.

**Ablaciones.** K ∈ {1, 2, 3, 5}; S ∈ {1, 4, 8}; fuente de candidatas (LLM, ASR, ambas); radio de agrupamiento; compuerta solo en t = 0 frente a cada replanteo; regla de arbitraje; en Fase B, un segundo VLA (π0.5 en GPU en la nube) y G-pert.

**Amenazas a la validez.** Atajos ciegos al lenguaje (control sin instrucción); fuga entre zonas (zonas reservadas); un solo operador (se documenta); **transferencia de morfología** (SmolVLA no vio xArm en el preentrenamiento: se reporta el éxito en inglés como techo y se discute); deriva de hardware (calibración diaria, orden aleatorio); sesgo del anotador (ocultamiento); comparaciones múltiples (Holm); brecha simulación-real (ODPB contra lazo cerrado).

### 2.4 Cronograma rebaselinado (al 3 de noviembre de 2026)

Estado de partida: documentos de propuesta y revisión listos; repositorio con solo la licencia; sin datos ni hardware integrado. El plan original de 10 semanas está en la semana 6. **Ninguna compuerta se ha pasado.**

| Bloque | Fechas | Contenido | Compuerta |
|---|---|---|---|
| A1 | 3–9 nov | Software base, driver xArm + supervisor, teleoperación, cámaras, piloto de LIBERO-Goal | — |
| A2 | 10–16 nov | Piloto de 50 demos de T1, primer ajuste de SmolVLA, controles | **G0:** éxito en inglés ≥ 50 % y ≥ 20 puntos sobre sin instrucción |
| A3 | 17–23 nov | Resto de demos, diagnóstico S0 en robot + ODPB, LIBERO-Goal en español | **G1:** español crudo y Spanglish cerca del azar y bajo inglés (p < 0.01, n ≥ 300 decisiones fuera de línea) |
| A4 | 24–30 nov | IMG y bases fuera de línea; demo para Expoingenierías | **G2:** IMG supera a toda base en AUROC en 2 de 3 cambios |
| A5 | 1–7 dic | Confirmación en robot (S1 vs S2, reducida), borrador v0, preprint opcional | **G3** (probablemente parcial) y **G4:** go/no-go del PM para la Fase B |
| B | 8 dic – 28 feb | n ≥ 100, S3 y S4, segundo VLA, T2, redacción | Envío a IROS 2027 o IEEE CASE 2027 (1 mar 2027) |

Este calendario es agresivo. Los cuellos de botella son la integración del xArm con LeRobot, la teleoperación y la recolección de demos. Una compuerta fallida activa el plan alterno de la escalera de afirmaciones, no una extensión silenciosa.

### 2.5 Presupuesto

El brazo y la laptop ya existen. Cifras en USD, sin envío ni aranceles; **estimaciones a verificar antes de comprar**.

| Concepto | Costo (USD, est.) | Notas |
|---|---|---|
| 2 cámaras USB | 30–60 | |
| Micrófono y bocina USB | 30–60 | |
| Objetos, charolas, sujetadores, pinzas de precisión para A1 | 30–60 | |
| Teleoperación: GELLO para xArm (servos Dynamixel + adaptador) | 250–350 | Alternativa: SpaceMouse, 150–200 |
| GPU en la nube (ajustes de SmolVLA y LIBERO) | 50–150 | SmolVLA, 20 mil pasos: unas 4 h en A100 según la documentación de LeRobot |
| **Núcleo** | **390–680** | |
| Pinza del xArm, si el laboratorio no la tiene | Por cotizar | Puede superar al resto del presupuesto |
| Fase B: GPU para π0.5 con LoRA (> 22.5 GB) | 100–250 | Se libera con G4 |

### 2.6 Riesgos

| # | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| R1 | SmolVLA no aprende bien en el xArm 6 (transferencia de morfología) o ignora el lenguaje | Media-alta | Alto | Control sin instrucción en G0; más demos; ACT como respaldo para el diagnóstico |
| R2 | Integración LeRobot–xArm toma más de una semana | Alta | Alto | Empezar el día 1; usar el SDK directo y un formato de dataset compatible |
| R3 | Teleoperación de baja calidad | Media | Alto | GELLO; evaluar SpaceMouse en el día 1 |
| R4 | No aparece brecha en español | Baja | Alto | G1; giro a vocabulario abierto u otro modelo |
| R5 | IMG no supera a las bases | Media | Medio | G2; reposicionar a diagnóstico y banco |
| R6 | Tiempo de robot (disponibilidad del xArm compartido) | Alta | Alto | Reservar bloques; ODPB y simulación en paralelo |
| R7 | Lesión o daño por un brazo de 5 kg de carga | Baja | Muy alto | Supervisor, modo reducido, paro físico, zona de exclusión, regla de dos personas |
| R8 | Otro grupo publica antes | Media | Medio | Preprint en G4; vigilancia semanal |
| R9 | Ética de datos de voz | Media | Medio | Consentimiento, anonimización; vía institucional definida por el PM |
| R10 | Traspaso a otro integrante del equipo | Media | Medio | `docs/HANDOFF.md` y `CLAUDE.md` en el repositorio |

### 2.7 Seguridad, ética y datos

**Seguridad física.** Protocolo escrito de paro antes de cada sesión; paro de emergencia al alcance; zona de exclusión marcada; modo reducido y límites del TCP en el controlador; nadie entra al espacio de trabajo durante un ensayo; regla de dos personas para cualquier ejecución autónoma; la primera ejecución de cada política nueva se hace a velocidad mínima. No se usa cautín encendido.

**Ética y voz.** Consentimiento informado, anonimización, almacenamiento cifrado y política de retención. La vía ética institucional es una decisión pendiente del PM. Solo se liberan grabaciones consentidas y sintéticas.

**Licencias.** Código: MIT (licencia del repositorio). Datos: CC-BY-4.0 propuesto. Verificar compatibilidad con SmolVLA, LeRobot, Whisper (MIT) y el LLM elegido.

### 2.8 Entregables y publicación

- **D1** Dataset de demostraciones (formato LeRobot) y banco de comandos.
- **D2** Arnés de evaluación y ODPB.
- **D3** Biblioteca IMG con envoltura para LeRobot y driver del xArm 6.
- **D4** Artículo y preprint.
- **D5** Video de demo y demo en vivo para Expoingenierías.
- **D6** Notas de montaje (cámaras, teleoperación, supervisor).

| Sede | Fecha límite | Comentario |
|---|---|---|
| IROS 2027 | 1 mar 2027 | Objetivo principal [27] |
| IEEE CASE 2027 | 1 mar 2027 | Alterno [27] |
| RA-L con opción IROS | Verificar 2027 | |
| IEEE AIM 2027 | 2 mar 2027 | Alterno [27] |
| ICRA 2027 | 15 sep 2026 (vencida) | No viable [28] |
| ACL/EMNLP Findings | Por confirmar | Para el análisis multilingüe |

### 2.9 Equipo y decisiones requeridas

| Rol | Responsabilidad |
|---|---|
| PM | Alcance, presupuesto, sede, vía ética |
| Líder técnico (Ian) | Hardware, datos, modelos, evaluación, redacción |
| Segundo integrante (por asignar) | Continuidad del proyecto según `docs/HANDOFF.md` |
| Dos anotadores nativos de español mexicano | Banco de comandos y etiquetado ciego |
| Asesor académico | Diseño experimental y estadística |

**Decisiones requeridas:** (1) presupuesto del núcleo; (2) horario de uso del xArm 6 y existencia de la pinza; (3) sede objetivo; (4) vía ética para voz; (5) créditos de GPU; (6) integración con el AGV.

---

## 3. Conclusiones

- El VLA rinde más en manipulación que en el AGV; ahí se concentra SABER.
- La afirmación central es medible y falsable: un VLA abierto y pequeño falla en silencio ante español, Spanglish y voz ruidosa, y las compuertas G0–G4 lo ponen a prueba.
- Con el xArm 6 se gana seguridad y repetibilidad y se pierde el argumento de bajo costo del brazo y la integración inmediata con LeRobot.
- La novedad es moderada: la diferencia frente a UPS es marginalizar sobre interpretaciones bajo cambio lingüístico, con una compuerta calibrada previa al movimiento y validada en robot real.
- El calendario es apretado. El riesgo dominante es la integración y la recolección de datos, no el método.

## 4. Pasos inmediatos

1. Confirmar con el PM y el laboratorio: horario del xArm 6, pinza, firmware y presupuesto.
2. Ejecutar el *speedrun* de software (prompt en `docs/SPEEDRUN_PROMPT.md`).
3. Elegir teleoperación (GELLO o SpaceMouse) y pedir piezas.
4. Congelar hipótesis y plan de análisis (`prereg/`) antes del primer ensayo real.
5. Redactar el banco de comandos con los dos hablantes nativos y preparar el consentimiento para voz.
6. Reproducir la línea base de LIBERO-Goal y agregar las traducciones al español.

---

## Apéndice A. Banco de comandos (muestra)

| Condición | Enunciado | Objetivo |
|---|---|---|
| E0 | "Pick up the pliers and put them in the tray" | Pinzas |
| E1 | "Grab the wrench and drop it in the tray" | Llave |
| S1 | "Toma la llave y ponla en la charola" | Llave |
| S2 | "Pásame el desarmador y ponlo en la charola" | Desarmador |
| S2 | "Échame la llave en la charola" | Llave |
| S3 | "Pásame el screwdriver" | Desarmador |
| S3 | "Agarra las pliers y ponlas en la tray" | Pinzas |
| A1 | "Pásame las pinzas" (dos tipos visibles) | Abstenerse y preguntar |
| O1 | "Pásame el multímetro" (ausente) | Abstenerse |

## Apéndice B. Objetos

| Español (variantes) | Inglés | Fase | Nota |
|---|---|---|---|
| Desarmador (destornillador) | Screwdriver | A | |
| Pinzas | Pliers | A | Ambiguas con pinzas de precisión (A1) |
| Llave | Wrench | A | Forma alargada |
| Flexómetro | Tape measure | B | |
| Cinta de aislar | Tape roll | B | Spanglish común: "el tape" |
| Carrete de estaño | Solder spool | B | Réplica, sin cautín |

Criterios: agarrables con la pinza del xArm, sin filos expuestos, buen contraste visual.

## Apéndice C. Métricas

Sea N el número de ensayos de una celda; N_ok los que colocan el objeto correcto; N_w los que agarran o se mueven hacia un objeto equivocado; N_f los que apuntan al correcto sin completar; N_a los que se abstienen antes de moverse.

- SR = N_ok / N · WOR = N_w / N · FCR = N_f / N · AR = N_a / N
- Cobertura = 1 − AR · Riesgo = N_w / (N − N_a)
- AURC: área bajo riesgo vs. cobertura al barrer τ [24]. AUROC: puntaje de la compuerta como discriminador de decisiones equivocadas frente a correctas.

## Apéndice D. Plan de análisis preregistrado

| Hipótesis | Prueba | Criterio |
|---|---|---|
| H1 | Binomial frente a sin instrucción; McNemar frente a inglés | α = 0.05, Holm |
| H2 | Wilson de la razón S1/E0; WOR residual | Razón ≥ 0.8 en S1; WOR ≥ 10 % en Spanglish o voz |
| H3 | AUROC con bootstrap por escena; diferencia con la mejor base | ≥ 0.03, IC 95 % sin cero, 2 de 3 cambios |
| H4 | McNemar sobre decisiones pareadas al 80 % de cobertura | Reducción relativa de WOR ≥ 50 % |
| H5 | Descriptiva | Exploratoria |

**Exclusiones.** Un ensayo con falla de hardware documentada antes de la decisión se repite y se reporta; nunca se descarta por su resultado. Orden aleatorizado por bloques; anotadores ciegos cuando sea posible; se reportan todas las condiciones, incluidas las negativas. El plan se registra en el repositorio (o en OSF) antes de los ensayos reales.

---

## Referencias

Las marcadas con † tienen título abreviado y autores omitidos; completar con arXiv antes de enviar.

1. K. Gubernatorov et al., "AnywhereVLA," arXiv:2509.21006, 2025.
2. X. Dong et al., "When does language matter? Multilingual instructions reveal step-wise language sensitivity in VLA models," ACL 2026, arXiv:2606.11906.
3. H. Chen et al., "Beyond English: Uncovering the multilingual gap in VLA models," arXiv:2606.15714, 2026.
4. A. J. Hancock et al., "Actions as language: Fine-tuning VLMs into VLAs without catastrophic forgetting," ICLR 2026, arXiv:2509.22195.
5. X. Zhou et al., "LIBERO-PRO," arXiv:2510.03827, 2025.
6. S. Fei et al., "LIBERO-Plus," arXiv:2510.13626, 2025.
7. J. Yuan et al., "When to act, ask, or learn: Uncertainty-aware policy steering," RSS 2026, arXiv:2602.22474.
8. A. Z. Ren et al., "Robots that ask for help," CoRL 2023.
9. R. Römer et al., "Failure prediction at runtime for generative robot policies," NeurIPS 2025, arXiv:2510.09459.
10. "SAFECAST," arXiv:2608.04246, 2026. †
11. T. Yang et al., "HiVLA," arXiv:2604.14125, 2026. †
12. A. Shaw et al., "Cutting the Cord," arXiv:2603.09051, 2026. †
13. uFactory, "xArm-Python-SDK" y "xarm_ros2" (URDF), repositorios de GitHub. Verificar versiones.
14. R. Cadene et al., "LeRobot," arXiv:2602.22818, 2026.
15. M. Shukor et al., "SmolVLA," arXiv:2506.01844, 2025.
16. A. Radford et al., "Robust speech recognition via large-scale weak supervision," ICML 2023.
17. L. Ben Allal et al., "SmolLM2," arXiv:2502.02737, 2025.
18. K. D. Nguyen et al., "vla.cpp," arXiv:2606.08094, 2026. †
19. L. Kuhn, Y. Gal, S. Farquhar, "Semantic uncertainty," ICLR 2023.
20. A. N. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction," arXiv:2107.07511, 2021.
21. "VLA-FAIL," arXiv:2606.21386, 2026. †
22. Y. Lee, D. Har, "Perturbation-based epistemic uncertainty for failure detection in VLA models," arXiv:2606.20754, 2026.
23. P. Wu et al., "GELLO: A general, low-cost, and intuitive teleoperation framework for robot manipulators," 2023 (verificar sede y soporte para xArm).
24. Y. Geifman, R. El-Yaniv, "Selective classification for deep neural networks," NeurIPS 2017.
25. B. Liu et al., "LIBERO," NeurIPS 2023.
26. D. Snyder et al., "Is your imitation learning policy better than mine?," RSS 2025, arXiv:2503.10966.
27. IEEE RAS, "Important dates: IROS 2027 and IEEE CASE 2027."
28. ICRA 2027, "Call for technical papers."
