# SABER: Documento de traspaso

Para: el integrante del equipo que continúa el proyecto, y para su asistente de IA.
De: Ian Carlos Hernández Robles · Última actualización: 3 de noviembre de 2026

> **Instrucciones para el asistente (Claude u otro):** este archivo es tu punto de partida. Antes de proponer o escribir nada, lee en este orden: (1) este documento completo, (2) `docs/PROPUESTA.md`, (3) `CLAUDE.md` y `docs/DECISIONS.md` si existen. La propuesta es la fuente de verdad del diseño experimental. No inventes resultados, parámetros del robot ni fechas; si algo no está aquí ni en la propuesta, pregunta al usuario. Toda ejecución en el robot real pasa por el supervisor de seguridad y requiere a un humano presente.

---

## 1. El proyecto en un párrafo

Los modelos visión-lenguaje-acción (VLA) abiertos se entrenan casi solo en inglés. Si les hablas en español no avisan que no entendieron: se mueven con seguridad hacia el objeto equivocado (en simulación, LIBERO-Goal, el éxito cae de 98 % a 11 %). SABER mide ese fallo silencioso en un **xArm 6 real** con una política **SmolVLA**, usando español formal, mexicano coloquial, Spanglish y voz ruidosa, y evalúa una **compuerta de verificación previa al movimiento (IMG)**: genera varias interpretaciones de la orden, revisa si todas llevan al mismo movimiento y, si no, el robot pregunta en español en lugar de adivinar. Meta: artículo en IROS 2027 o IEEE CASE 2027 (fecha límite: 1 de marzo de 2027).

## 2. Documentos y dónde están

| Archivo | Qué es | Estado |
|---|---|---|
| `docs/PROPUESTA.md` | Propuesta v2 (xArm 6). Diseño experimental completo | Vigente |
| `SABER_Propuesta.pdf` | Propuesta v1 (SO-101) | **Obsoleta** en hardware, presupuesto y cronograma |
| `SABER_Revision_Literatura.pdf` | Revisión de literatura 2022–2026 y matriz de brechas | Vigente; hay entradas † por completar |
| `SABER_Slide.pptx` | Lámina de resumen | Desactualizada: dice SO-101 y USD 400–750, y no aclara que 98 % → 11 % es simulación |
| `docs/SPEEDRUN_PROMPT.md` | Prompt para que Claude Code construya el software | Usar una vez; luego revisar resultados |
| `CLAUDE.md`, `docs/DECISIONS.md`, `docs/SAFETY.md`, `docs/SETUP.md` | Los crea el speedrun | Ver sección 5 |

Repositorio: `github.com/ICHR-BME/SABER` (licencia MIT).

## 3. Decisiones ya tomadas (no reabrir sin motivo)

1. El VLA va en **manipulación de mesa**, no en el AGV.
2. Brazo: **uFactory xArm 6** del laboratorio (no SO-101). Consecuencias en la sección 0 de la propuesta.
3. Política principal: **SmolVLA**, ajustada solo con instrucciones en inglés (eso es lo que se diagnostica).
4. Fase A con **3 objetos**: desarmador, pinzas, llave. Tarea T1: "toma X y ponlo en la charola".
5. La novedad declarada es **moderada**. El antecedente más cercano es UPS (Yuan et al., RSS 2026). No vender la compuerta como "primera" ni la brecha multilingüe como "nunca medida".
6. Las cifras de LIBERO son **de simulación**; decirlo siempre.
7. Código bajo MIT.

## 4. Decisiones abiertas (resolver primero)

| # | Decisión | Quién | Bloquea |
|---|---|---|---|
| 1 | Horario de uso del xArm 6 y quién más lo usa | PM / laboratorio | Todo el trabajo en robot |
| 2 | ¿Hay pinza para el xArm? ¿Qué modelo? | Laboratorio | Demos |
| 3 | ¿Es xArm 6 o Lite 6? (son robots distintos) | Confirmar físicamente | Driver, URDF, abstract de la expo |
| 4 | Teleoperación: GELLO (~USD 250–350) o SpaceMouse (~USD 150–200) | Equipo + PM | Demos |
| 5 | Presupuesto del núcleo (~USD 390–680) | PM | Compras |
| 6 | Vía ética para grabaciones de voz | PM | Condiciones V1 y V2 |
| 7 | Sede final: IROS o CASE | PM | Formato del artículo |
| 8 | Quiénes son los dos hablantes nativos que revisan el banco de comandos y anotan videos | Equipo | Condiciones S2, S3 y anotación ciega |
| 9 | Fecha de Expoingenierías y qué se muestra | Equipo | Demo D5 |

## 5. Estado del software

Al 3 de noviembre el repositorio solo contenía `LICENSE`. El speedrun de Claude Code (`docs/SPEEDRUN_PROMPT.md`) debe llenar esta tabla al terminar. **Si sigue vacía, el speedrun no se ha corrido o no terminó.** Verifica cada fila con su prueba; no confíes en la palabra "hecho".

| Módulo | Estado (hecho / parcial / stub / pendiente) | Archivo | Prueba que lo demuestra |
|---|---|---|---|
| Esqueleto, CI, `CLAUDE.md` | pendiente | | |
| Supervisor de seguridad | pendiente | | |
| Driver xArm 6 + `MockXArm` | pendiente | | |
| Cinemática desde URDF oficial | pendiente | | |
| Teleoperación (GELLO / SpaceMouse) | pendiente | | |
| Grabación a LeRobotDataset | pendiente | | |
| Banco de comandos (borrador) | pendiente | | |
| ASR n-best + mezcla de ruido | pendiente | | |
| Intérprete K candidatas | pendiente | | |
| Envolturas SmolVLA / ACT / analizador | pendiente | | |
| IMG + compuertas base + calibración | pendiente | | |
| ODPB, métricas, estadística, simulador de potencia | pendiente | | |
| Generador de ensayos y hojas de captura | pendiente | | |
| LIBERO-Goal en español | pendiente | | |
| Preregistro | pendiente | | |

## 6. Lo que tiene que hacer un humano (Claude no puede)

### 6.1 Esta semana (3–9 nov)
- [ ] Resolver las decisiones abiertas 1–5.
- [ ] Correr el speedrun y revisar su reporte; correr `uv run pytest` tú mismo.
- [ ] Ir al laboratorio: confirmar modelo del brazo, pinza, IP del controlador y firmware.
- [ ] Probar `scripts/xarm_check.py` **con alguien presente y el paro de emergencia en la mano**.
- [ ] Montar las dos cámaras (imprimir soporte de muñeca) y verificar que se graban a la misma frecuencia.
- [ ] Pedir las piezas de teleoperación.

### 6.2 Semana 10–16 nov (meta G0)
- [ ] Armar la escena: tres objetos, charola, cinco zonas marcadas con cinta.
- [ ] Grabar el **piloto de 50 demos** de T1, instrucciones solo en inglés.
- [ ] Rentar GPU y ajustar SmolVLA; correr el piloto en el robot con E0 y sin instrucción.
- [ ] **G0:** ¿éxito en inglés ≥ 50 % y ≥ 20 puntos sobre sin instrucción? Si no: más demos o revisar R1 de la propuesta. No avanzar sin G0.
- [ ] Congelar el preregistro (hash del plan) **antes** de cualquier ensayo con español.

### 6.3 Semana 17–23 nov (meta G1)
- [ ] Completar 150–225 demos (incluye 20 % en zonas reservadas).
- [ ] Revisar con los hablantes nativos el banco de comandos (cambiar `draft_needs_native_review` a final).
- [ ] Grabar voz V1 y V2 **con consentimiento firmado**.
- [ ] Correr S0 en robot y construir el ODPB.
- [ ] **G1:** ¿español crudo y Spanglish cerca del azar y significativamente bajo inglés?

### 6.4 Semana 24–30 nov (meta G2 y expo)
- [ ] Correr IMG y compuertas base sobre el ODPB.
- [ ] **G2:** ¿IMG supera a todas las bases en AUROC en 2 de 3 tipos de cambio? Si no, C3 se retira y el artículo se vuelve diagnóstico + banco.
- [ ] Preparar demo de expo: una orden en inglés que funciona, una en español que falla sin compuerta y la misma con compuerta preguntando.

### 6.5 Semana 1–7 dic
- [ ] Confirmación reducida en robot: S1 vs S2.
- [ ] Borrador v0 del artículo y video.
- [ ] **G4:** reunión con el PM para decidir la Fase B.

### 6.6 Fase B (8 dic – 28 feb)
- [ ] Subir celdas clave a n ≥ 100; correr S3 y S4.
- [ ] Segundo VLA (π0.5 en la nube) y G-pert.
- [ ] T2 y los otros tres objetos si hay tiempo.
- [ ] Completar referencias † y leer completos UPS, SAFECAST, VLA-FAIL, Dong et al. y Chen et al.
- [ ] Envío el 1 de marzo de 2027.

## 7. Reglas de seguridad con el xArm 6

Resumen; la versión completa está en `docs/SAFETY.md`.
- Nunca solo: dos personas en cualquier ejecución autónoma.
- Paro de emergencia en la mano, zona de exclusión marcada, nadie dentro durante un ensayo.
- Modo reducido y caja del TCP activos en el controlador, además del supervisor en software.
- Toda política nueva se prueba primero a velocidad mínima.
- El sistema no está certificado (ISO 10218, ISO/TS 15066); no lo describas como seguro para colaboración.

## 8. Errores que no hay que cometer

- Reportar cifras de simulación como si fueran de robot real.
- Mirar los resultados en español antes de congelar el preregistro.
- Descartar ensayos por su resultado. Solo se repiten fallas de hardware documentadas **antes** de la decisión.
- Usar el modo de guiado a mano del xArm para grabar demos: la mano del operador aparece en la cámara.
- Mezclar zonas de entrenamiento y evaluación.
- Dejar que el asistente "complete" números faltantes o parámetros del robot.
- Seguir con un plan fallido sin pasar por la compuerta correspondiente.

## 9. Cómo trabajar con tu asistente

1. Abre Claude Code en la raíz del repositorio. Si `CLAUDE.md` existe, se carga solo; si no, pídele que lea este archivo primero.
2. Para conversación general (planeación, redacción), crea un Project en Claude y sube `docs/PROPUESTA.md`, este archivo y la revisión de literatura.
3. Pide tareas acotadas y verificables ("implementa X y su prueba"), no "termina el proyecto".
4. Al final de cada sesión, pídele que actualice la sección 5 de este archivo y `docs/DECISIONS.md`.
5. Cuando cambie algo del diseño, actualiza `docs/PROPUESTA.md` y anota el motivo en `docs/DECISIONS.md`.

## 10. Registro en Expoingenierías

Textos ya preparados en el chat de Ian (versión xArm 6):
- **Nombre:** "SABER: A Robot Arm That Verifies Spanish Commands Before Acting"
- **Objetivo:** "Measure and prevent silent errors of AI robots given Spanish commands: verify the command's meaning before moving and ask for clarification instead of guessing."
- **Abstract:** cinco secciones, unas 244 palabras (límite 250). Pedírselo a Ian si no se ha enviado.
- **Materia:** MR-2035. Falta confirmar profesor coordinador y número de integrantes.

## 11. Glosario

- **VLA:** modelo visión-lenguaje-acción; recibe imagen + texto y produce acciones del robot.
- **SmolVLA:** VLA abierto de 450 M parámetros de Hugging Face.
- **IMG:** Compuerta de Interpretación Marginalizada, el método del proyecto.
- **ODPB:** banco de puntos de decisión fuera de línea; evalúa compuertas sin usar el robot.
- **WOR:** tasa de objeto equivocado. **AR:** tasa de abstención. **Cobertura:** 1 − AR.
- **G0–G4:** compuertas de decisión go/no-go del cronograma.
- **C1–C4:** afirmaciones del artículo, de más a menos segura.
- **LIBERO-Goal:** banco de simulación donde la instrucción es la única pista de la tarea.
