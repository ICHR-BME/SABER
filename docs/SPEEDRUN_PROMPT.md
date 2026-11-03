# Prompt de speedrun para Claude Code

Copia todo lo que está debajo de la línea y pégalo en Claude Code abierto en la raíz del repositorio `ICHR-BME/SABER`. Antes, copia `docs/PROPUESTA.md` (v2) al repositorio.

---

Eres el ingeniero de software principal del proyecto SABER. Tu trabajo en esta sesión es construir **todo el software que no requiere hardware ni datos reales**, dejándolo probado, documentado y listo para que un humano conecte el xArm 6, grabe demostraciones y corra experimentos.

## Contexto obligatorio

1. Lee completo `docs/PROPUESTA.md`. Es la fuente de verdad para tareas, condiciones de lenguaje, sistemas S0–S4, compuertas base, métricas, hipótesis y plan de análisis. Si algo de este prompt contradice la propuesta, gana la propuesta y lo anotas en `docs/DECISIONS.md`.
2. El repositorio hoy solo tiene `LICENSE` (MIT). Todo lo demás lo creas tú.
3. Hardware objetivo: uFactory xArm 6 con pinza, dos cámaras USB (muñeca y frontal), laptop con RTX 5060 para inferencia. **No tienes acceso al robot.** Todo debe funcionar con un robot simulado (`MockXArm`) y con datos sintéticos.

## Reglas no negociables

- **Nunca inventes resultados.** No escribas números de éxito, AUROC ni latencias en documentos como si fueran medidos. Los fixtures sintéticos se marcan como sintéticos en el nombre del archivo y en el código.
- **Nunca inventes parámetros del robot.** La cinemática del xArm 6 sale del URDF oficial de uFactory (repositorio `xarm_ros2`) o de la función de cinemática directa del `xArm-Python-SDK`. Si no puedes descargarlos, deja la interfaz con un `NotImplementedError` claro y una prueba marcada `skip`, y regístralo en `docs/HANDOFF.md`.
- **Nunca escribas código que mueva el robot real sin pasar por el supervisor de seguridad.** El driver real debe exigir un flag explícito (`--i-am-supervising`) y arrancar en modo reducido.
- **Verifica APIs antes de usarlas.** Para LeRobot, SmolVLA, xArm-Python-SDK, Whisper y LIBERO: instala la versión, inspecciona el código o la documentación y fija versiones en `pyproject.toml`. Si una API no existe como esperabas, adáptate y anótalo en `docs/DECISIONS.md`.
- Haz un commit por fase con mensaje claro. **No hagas push** sin que el usuario lo pida.
- Si una fase te bloquea más de ~30 minutos de intentos, déjala con stubs documentados y sigue con la siguiente.
- Código, comentarios y docstrings en inglés. `docs/HANDOFF.md` en español.

## Estructura objetivo

```
saber/
  robots/xarm6/      driver LeRobot-compatible, MockXArm, cinemática, config
  safety/            supervisor (independiente del modelo)
  teleop/            adaptadores GELLO, SpaceMouse, teclado → acciones
  record/            grabación a LeRobotDataset (o formato compatible)
  lang/              banco de comandos, ASR n-best, intérprete K candidatas
  gate/              IMG, compuertas base, calibración conformal
  policies/          envolturas SmolVLA y ACT con interfaz común
  eval/              ODPB, métricas, estadística, aleatorización, anotación
  sim/libero/        traducción de LIBERO-Goal y script de evaluación
configs/             YAML por experimento (S0–S4)
prereg/              hipótesis y plan de análisis congelables
scripts/             CLIs
tests/
docs/                PROPUESTA.md, DECISIONS.md, HANDOFF.md, SAFETY.md, SETUP.md
CLAUDE.md
```

## Fases

Crea una lista de tareas con estas fases y márcalas al avanzar. Cada fase termina con `pytest` en verde.

### Fase 0. Esqueleto
- `pyproject.toml` (Python ≥ 3.10, gestor `uv`), `ruff`, `pytest`, GitHub Action que corre lint y pruebas sin GPU.
- `CLAUDE.md` con: qué es el proyecto, comandos para instalar y probar, mapa del repositorio, las reglas no negociables de arriba y "lee `docs/HANDOFF.md` antes de trabajar".
- **Aceptación:** `uv sync && uv run pytest` pasa en una máquina sin GPU.

### Fase 1. Seguridad y driver del xArm 6
- `safety/supervisor.py`: límites articulares, caja cartesiana del TCP, tope de velocidad y aceleración por paso, rechazo de saltos, watchdog de latencia y de cámara, paro. Configurable por YAML. Pruebas unitarias exhaustivas, incluidas propiedades (hypothesis): ninguna acción que sale del supervisor viola un límite.
- `robots/xarm6/`: clase `Robot` compatible con la interfaz de la versión fijada de LeRobot, sobre el xArm-Python-SDK. Espacio de acción: 6 articulaciones + pinza. Al conectar: modo reducido, sensibilidad de colisión alta y límites del TCP en el controlador (verifica los nombres de las funciones en el SDK instalado).
- `MockXArm` con la misma interfaz, cinemática real y cámaras sintéticas.
- `scripts/xarm_check.py`: conecta, lee estado, mueve una articulación 2° a velocidad mínima solo con `--i-am-supervising`.
- `docs/SAFETY.md`: protocolo de sesión (checklist previo, paro, zona de exclusión, regla de dos personas).

### Fase 2. Teleoperación y grabación
- Adaptadores de teleoperación con interfaz común: GELLO (mapeo articular), SpaceMouse (cartesiano + IK) y teclado (solo para depuración).
- `record/`: grabación de episodios con ambas cámaras, estado, acción e instrucción en inglés, a `LeRobotDataset`. Metadatos por episodio: objetivo, zona, posiciones de los tres objetos, operador y split (train / zona reservada).
- `scripts/record_demos.py` con protocolo guiado: le dice al operador qué objetivo y qué zona toca según una asignación balanceada.
- Prueba de extremo a extremo con `MockXArm` y teleoperación sintética.

### Fase 3. Lenguaje
- `lang/command_bank/`: YAML con las condiciones E0, E1, S1, S2, S3, A1 y O1 para los tres objetos de la Fase A. Redacta borradores de 30 enunciados por condición, **marcados `status: draft_needs_native_review`**. No los marques como finales.
- `lang/asr.py`: Whisper con n-best (beam o temperatura) y log-probabilidades; wrapper de faster-whisper si es más rápido. Mezclador de ruido con SNR configurable para V2.
- `lang/interpreter.py`: interfaz para muestrear K instrucciones canónicas en inglés con pesos, a partir del enunciado y opcionalmente el fotograma. Backends: modelo local de Hugging Face y un backend determinista de prueba. Script del piloto de 200 enunciados para comparar candidatos de intérprete.

### Fase 4. Políticas
- `policies/smolvla.py`: carga, inferencia por trozos, muestreo de S trozos por instrucción reutilizando el prefijo si la API lo permite (si no, documenta el costo real).
- `policies/act.py` y `policies/parser_act.py` para el sistema S4 (analizador de reglas sobre el vocabulario fijo + una política ACT por objeto).
- Scripts de ajuste fino para correr en GPU rentada, con configs y estimación de horas. No los ejecutes.
- Script de inferencia en vivo con supervisor, compuerta opcional y registro completo de cada decisión.

### Fase 5. Compuerta IMG y bases
- `gate/img.py`: exactamente lo de la sección 2.2.2 de la propuesta. Característica φ por cinemática directa, agrupamiento con radio relativo a la dispersión intracandidata, m*, H_B, R, arbitraje por grupo modal, pregunta de aclaración en español con los dos objetivos más probables.
- `gate/baselines.py`: G-text, G-conf, G-ood, G-act y G-int (G-pert como stub de Fase B).
- `gate/calibrate.py`: control de riesgo de estilo conformal dividido para fijar τ al nivel α.
- Pruebas con distribuciones sintéticas donde el resultado correcto se conoce: candidatas que coinciden → ejecuta; candidatas que apuntan a objetos distintos → se abstiene; garantía de riesgo empírica en simulación de Monte Carlo.

### Fase 6. Evaluación y estadística
- `eval/odpb.py`: construye el banco de puntos de decisión desde episodios reservados y cualquier política o compuerta.
- `eval/metrics.py`: SR, WOR, FCR, AR, riesgo, cobertura, AURC y AUROC (Apéndice C).
- `eval/stats.py`: Wilson, McNemar exacto, bootstrap por conglomerados de escena, Holm, kappa de Cohen, y un **simulador de potencia** que verifique la afirmación de "40 ensayos pareados ≈ 30 puntos con 80 %". Reporta el resultado en `docs/DECISIONS.md`, sea cual sea.
- `eval/trials.py`: genera la lista de ensayos de S0–S4 aleatorizada por bloques, con escenas pareadas, y una hoja de captura (CSV) para el operador y otra para anotadores ciegos.
- `scripts/report.py`: tablas y figuras desde los CSV. Con datos sintéticos, cada figura lleva la marca "SYNTHETIC".

### Fase 7. Simulación LIBERO-Goal
- Traducción de las instrucciones de LIBERO-Goal a español y otros tres idiomas con un LLM, revisión marcada como pendiente.
- Script de evaluación de S0, S1 y S2 con un punto de control de SmolVLA para LIBERO si existe (búscalo y verifícalo); si no, config de ajuste propio.
- Debe poder correr en una GPU rentada con un solo comando. No lo ejecutes si no hay GPU.

### Fase 8. Preregistro y traspaso
- `prereg/hypotheses.md` y `prereg/analysis_plan.yaml` con H1–H5, pruebas y criterios del Apéndice D, más un script que calcula el hash del plan para congelarlo.
- `docs/SETUP.md`: instalación desde cero, conexión del xArm, cámaras, teleoperación.
- Actualiza `docs/HANDOFF.md` (ya existe; no lo reescribas desde cero): completa la tabla "Estado del software" con lo que quedó **hecho, parcial o stub**, con el archivo y la prueba que lo demuestran, y agrega cualquier tarea humana nueva que descubras.

## Al terminar

Reporta en un mensaje corto:
1. Qué fases quedaron completas, parciales o en stub.
2. Versiones fijadas y cualquier API que no existía como se esperaba.
3. Las tres cosas que un humano debe hacer primero con el robot.
4. Comandos exactos para verificar todo (`uv run pytest`, etc.).
