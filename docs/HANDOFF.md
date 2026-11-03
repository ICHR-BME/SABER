# SABER: Documento de traspaso

Para: el integrante del equipo que continúa el proyecto, y para su asistente de IA.
De: Ian Carlos Hernández Robles · Última actualización: 3 de noviembre de 2026 (tabla de la sección 5 y tareas 6.0 llenadas por el speedrun)

> **Instrucciones para el asistente (Claude u otro):** este archivo es tu punto de partida. Antes de proponer o escribir nada, lee en este orden: (1) este documento completo, (2) `docs/PROPUESTA.md`, (3) `CLAUDE.md` y `docs/DECISIONS.md` si existen. La propuesta es la fuente de verdad del diseño experimental. No inventes resultados, parámetros del robot ni fechas; si algo no está aquí ni en la propuesta, pregunta al usuario. Toda ejecución en el robot real requiere a un humano presente con el paro de la celda a la mano.

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
| `CLAUDE.md`, `docs/DECISIONS.md`, `docs/SAFETY.md`, `docs/SETUP.md` | Guía del repo, decisiones y APIs verificadas, checklist de seguridad, instalación | Creados por el speedrun (3 nov 2026) |
| `prereg/` | Hipótesis y plan de análisis | Borrador, **no congelado** |

Repositorio: `github.com/ICHR-BME/SABER` (licencia MIT).

## 3. Decisiones ya tomadas (no reabrir sin motivo)

1. Lo importante es el **VLA**. El brazo es fijo y se prueba en una **celda**. Nada de base móvil ni AGV.
2. Brazo: **uFactory xArm 6** (confirmado, no Lite 6), fijo en la celda. Se controla **directo por IP** con el xArm-Python-SDK; no hay otra capa de control de la celda en medio.
2b. Cámaras: las fijas de la celda + **una cámara de muñeca** que se va a montar.
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
| 4 | Teleoperación: GELLO (~USD 250–350) o SpaceMouse (~USD 150–200) | Equipo + PM | Demos |
| 5 | Presupuesto del núcleo (~USD 390–680) | PM | Compras |
| 6 | Vía ética para grabaciones de voz | PM | Condiciones V1 y V2 |
| 7 | Sede final: IROS o CASE | PM | Formato del artículo |
| 8 | Quiénes son los dos hablantes nativos que revisan el banco de comandos y anotan videos | Equipo | Condiciones S2, S3 y anotación ciega |
| 9 | Fecha de Expoingenierías y qué se muestra | Equipo | Demo D5 |
| 10 | IP del controlador y resolución/frecuencia de las cámaras fijas de la celda | Laboratorio | Configuración |

## 5. Estado del software

Al 3 de noviembre el repositorio solo contenía `LICENSE`. El speedrun de Claude Code (`docs/SPEEDRUN_PROMPT.md`) se corrió el 3 de noviembre de 2026 y llenó esta tabla. Verifica cada fila con su prueba; no confíes en la palabra "hecho". **Todo se probó solo con `MockXArm` y datos sintéticos, en CPU: nada se ha corrido contra el robot, una GPU, un modelo descargado ni LIBERO.** Los detalles y desviaciones están en `docs/DECISIONS.md`.

"Hecho" = implementado y probado sin hardware. "Parcial" = implementado, pero su parte principal solo se puede verificar con hardware/GPU/modelos. "Stub" = interfaz sin implementación.

| Módulo | Estado (hecho / parcial / stub / pendiente) | Archivo | Prueba que lo demuestra |
|---|---|---|---|
| Esqueleto, CI, `CLAUDE.md` | hecho | `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `CLAUDE.md` | `uv sync && uv run pytest` (76 pasan, 3 se saltan sin extras) |
| Límites mínimos (recorte de acciones, paro) | hecho | `saber/safety/limits.py`, `configs/safety/xarm6_minimal.yaml` | `tests/test_safety_cameras.py` |
| Driver xArm 6 + `MockXArm` | parcial: probado contra un SDK simulado con los mismos nombres de función; falta probar en el brazo | `saber/robots/xarm6/driver.py`, `mock.py`, `lerobot_compat.py`, `scripts/xarm_check.py` | `tests/test_xarm6.py` |
| Cinemática desde URDF oficial | hecho (parámetros copiados de `xarm_ros2` @ `ac736c9`; verificar contra la FK del controlador con `xarm_check.py`) | `saber/robots/xarm6/kinematics.py`, `data/xarm6_official.yaml` | `tests/test_xarm6.py::test_fk_zero_pose_matches_known_flange_position` |
| Cámaras (muñeca + celda, sincronización) | parcial: lector OpenCV/RTSP sin probar con cámaras reales | `saber/cameras/`, `configs/cameras/cell.yaml` | `tests/test_safety_cameras.py` |
| Teleoperación (GELLO / SpaceMouse) | parcial: mapeos probados con lectores falsos; sin hardware | `saber/teleop/` | `tests/test_teleop_record.py` |
| Grabación a LeRobotDataset | parcial: npz compatible probado de extremo a extremo; escritor `LeRobotDataset` sin correr (necesita el extra `lerobot`) | `saber/record/`, `scripts/record_demos.py` | `tests/test_teleop_record.py::test_end_to_end_synthetic_recording` |
| Banco de comandos (borrador) | hecho como borrador: 183 enunciados, todos `draft_needs_native_review` | `saber/lang/command_bank/phase_a.yaml` | `tests/test_lang.py::test_bank_complete_and_all_drafts` |
| ASR n-best + mezcla de ruido | parcial: n-best probado con un Whisper falso; mezclador exacto | `saber/lang/asr.py` | `tests/test_lang.py` |
| Intérprete K candidatas | parcial: backend de reglas probado; backend Hugging Face sin correr; piloto listo | `saber/lang/interpreter.py`, `scripts/interpreter_pilot.py` | `tests/test_lang.py` |
| Envolturas SmolVLA / ACT / analizador | parcial: analizador + S4 probados; SmolVLA (prefijo compartido) y ACT escritos contra el código de lerobot 0.4.4 pero **sin ejecutar** | `saber/policies/`, `saber/live.py`, `scripts/run_live.py`, `scripts/train_policy.py` | `tests/test_policies.py` |
| IMG + compuertas base + calibración | hecho (G-pert: stub de Fase B) | `saber/gate/` | `tests/test_gate.py` |
| ODPB, métricas, estadística, simulador de potencia | hecho (probado con datos sintéticos) | `saber/eval/`, `scripts/build_odpb.py`, `calibrate_gate.py`, `power_sim.py`, `report.py` | `tests/test_eval.py` |
| Generador de ensayos y hojas de captura | hecho (800 ensayos, hoja ciega para anotadores) | `saber/eval/trials.py`, `scripts/make_trials.py` | `tests/test_eval.py::test_trials_total_pairing_and_blocks` |
| LIBERO-Goal en español | parcial: traducciones (es/pt/fr/de, borrador) y evaluador escritos; **sin ejecutar** (necesita Linux + GPU) | `saber/sim/libero/`, `scripts/eval_libero.py`, `scripts/gpu_libero.sh` | `tests/test_libero.py` |
| Preregistro | hecho como borrador, **no congelado** (tiene `open_items`) | `prereg/`, `scripts/freeze_prereg.py` | `tests/test_prereg.py` |

## 6. Lo que tiene que hacer un humano (Claude no puede)

### 6.0 Tareas nuevas que encontró el speedrun (3 nov)
- [ ] **Medir y llenar los valores de la celda** (hoy en `null` a propósito; el driver y la grabación se niegan a correr sin ellos): IP, posiciones abierta/cerrada de la pinza, sensibilidad de colisión, velocidad TCP reducida y caja del TCP en `configs/robot/xarm6_cell.yaml`; centros de las cinco zonas y de la charola en `configs/scene/phase_a.yaml` (las coordenadas actuales son sintéticas).
- [ ] Confirmar con `xarm_check.py` que la FK del URDF y la FK del controlador coinciden con la pinza montada (offset del TCP = 0.172 m según el URDF de la xArm Gripper).
- [ ] En la laptop: `uv sync --extra lerobot` y comprobar que `torch.cuda.is_available()` es `True` con la RTX 5060 (se instala torch 2.10 + cu128).
- [ ] Correr una vez SmolVLA con `SmolVLAChunkPolicy` en GPU: verificar que el muestreo con prefijo compartido da trozos con la misma distribución que `predict_action_chunk` y medir la latencia real (no hay ninguna latencia medida todavía).
- [ ] **Potencia:** el simulador muestra que 40 pares detectan 30 puntos con 80 % solo si los resultados de los dos sistemas están correlacionados; con independencia la potencia baja a 0.70–0.74 y hacen falta 46–50 pares (`docs/DECISIONS.md`, D7). Decidir n por celda antes de G0.
- [ ] **H2** pide español formal (condición S1), pero los ensayos reales de S1 son E0, S2, S3, V1 y V2: agregar una celda o evaluar esa parte de H2 en el ODPB.
- [ ] Elegir las dos SNR de V2 (el generador usa 10 dB y 0 dB como propuesta) y resolver los demás `open_items` de `prereg/analysis_plan.yaml`; después congelar con `uv run python scripts/freeze_prereg.py --freeze`.
- [ ] Correr el piloto del intérprete (`scripts/interpreter_pilot.py --config configs/interp/pilot_candidates.yaml`, necesita GPU y descargar modelos; Gemma y Llama piden aceptar licencia).
- [ ] Revisión nativa de: el banco de comandos, `train_paraphrases.yaml` (S3) y las traducciones de LIBERO-Goal.
- [ ] Confirmar el modelo de servos del GELLO (el código asume Dynamixel serie X).
- [ ] En una GPU rentada: `bash scripts/gpu_libero.sh` y revisar si `lerobot/smolvla_libero` sirve para LIBERO-Goal (no se verificó su desempeño).

### 6.1 Esta semana (3–9 nov)
- [ ] Resolver las decisiones abiertas 1–5.
- [ ] Correr el speedrun y revisar su reporte; correr `uv run pytest` tú mismo.
- [ ] Ir al laboratorio: confirmar modelo del brazo, pinza, IP del controlador y firmware.
- [ ] Probar `scripts/xarm_check.py` **con alguien presente y el paro de emergencia en la mano**.
- [ ] Imprimir el soporte y montar la cámara de muñeca; verificar que se lee junto con las cámaras fijas de la celda y se graban sincronizadas.
- [ ] Conocer el procedimiento de seguridad de la celda (paro, reinicio, acceso) antes de la primera ejecución.
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
- Se siguen las reglas de seguridad de la celda. Paro de emergencia de la celda en la mano; nadie dentro durante un ensayo.
- Al conectar, dejar activos en el controlador la detección de colisión, la velocidad reducida y la caja del TCP. Es el mínimo para que un error del VLA no choque el brazo contra la mesa o la cámara de muñeca.
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
