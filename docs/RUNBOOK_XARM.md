# Runbook: sesiones con el xArm 6 en la celda

Para integrantes del equipo que no escribieron el código. Lee antes `docs/SAFETY.md`. Todos los
comandos se corren desde la raíz del repositorio en la laptop. Nada de lo que aparece aquí se ha
probado todavía contra el brazo real: los scripts se probaron solo con SDKs simulados.

Datos conocidos de la celda (5 nov 2026): controlador en `192.168.1.226`, máscara `255.255.255.0`.
**Todo lo demás está sin medir** (pinza, sensibilidad de colisión, velocidad reducida, caja del TCP,
pose de casa, cámaras, zonas). No lo llenes con valores "típicos".

---

## 1. Roles en cada sesión

Mínimo **dos personas**; se recomiendan **tres**.

| Rol | Qué hace | Qué no hace |
|---|---|---|
| **Operador del paro** | Sostiene el paro de emergencia de la celda todo el tiempo que el brazo tenga motores habilitados. Mira el brazo, no la pantalla. Presiona sin dudar. | No teclea, no usa la laptop, no entra a la celda. |
| **Operador de laptop** | Corre los comandos, lee la salida en voz alta y decide si se sigue al siguiente peldaño. | No mueve el brazo sin avisar ("voy a mover la articulación 6"). |
| **Registrador** | Llena la bitácora de la sesión (sección 4) mientras pasa todo. Con dos personas, lo hace el operador de laptop. | — |

Nadie dentro del espacio de trabajo mientras haya motores habilitados.

## 2. Preparar la laptop (una sola vez)

1. Instalar el extra del robot:

   ```bash
   uv sync --extra robot
   ```

   (En Windows, si `uv` no está en el PATH: `%APPDATA%\Python\Python311\Scripts\uv.exe`.)
2. Conectar un cable Ethernet de la laptop al controlador del xArm.
3. Poner una **IP fija** en el adaptador Ethernet dentro de `192.168.1.0/24`, elegida **con el
   laboratorio** (no `.203`, que es el controlador), **sin puerta de enlace**. Antes de asignarla,
   comprueba que esté libre: `ping 192.168.1.<x>` **no** debe contestar. El script `net_check.py`
   imprime el comando exacto, pero nunca lo corre. En Windows (PowerShell como administrador):

   ```bash
   netsh interface ipv4 set address name="Ethernet" static 192.168.1.<x> 255.255.255.0
   ```

   En Linux: `sudo ip addr add 192.168.1.<x>/24 dev <dispositivo>`.
4. **Wi-Fi:** apágalo, o asegúrate de que la red Wi-Fi **no** sea `192.168.1.x` (muchos routers usan
   esa red y Windows mandaría el tráfico del robot por el Wi-Fi). Desactiva también VPNs (por
   ejemplo Cloudflare WARP) durante la sesión.
5. **Firewall de Windows:** el SDK solo abre conexiones de salida (TCP 502 y 30002), que Windows
   permite por defecto. Si un antivirus o suite de seguridad las bloquea, agrega una excepción.
6. Probar: `ping 192.168.1.226`.

## 3. Escalera de pruebas

**No te saltes peldaños. Cada sesión empieza otra vez en el peldaño 1.** Si se cumple un "Para si",
detente, anótalo en la bitácora y resuélvelo antes de seguir.

### Peldaño 1. Red

```bash
uv run python scripts/net_check.py
```

- **Esperado:** todas las líneas `[OK  ]` en lo requerido y al final `RESULT: all checks passed`
  (código de salida 0). Las líneas `[WARN]` de puertos opcionales (30001, 30003, 503, 18333) no
  bloquean.
- **Para si:** `FAIL` en "exactly one local interface", "Wi-Fi and Ethernet", o en los puertos 502 o
  30002.

### Peldaño 2. Sondeo de solo lectura (sin movimiento)

```bash
uv run --extra robot python scripts/xarm_check.py
```

Solo lee: no habilita motores, no limpia errores, no cambia modo ni estado. Guarda
`outputs/sessions/<fecha>/probe_<hora>.json` (con el commit de git y el hash de la configuración).

- **Esperado:** `error_code=0  warn_code=0`; firmware y número de serie impresos; "TCP (mm,
  controller)" y "TCP (mm, URDF FK with tcp_offset_m)" parecidos (pocos mm); ninguna línea `WARNING`.
- **Para si:** `error_code` o `warn_code` distintos de 0 (ver sección 5); aparece
  `WARNING: controller TCP offset ... NOT comparable` (el offset del TCP en el controlador no es el
  de `tcp_offset_m` = 0.172 m en z: revisa en UFactory Studio qué herramienta/offset está
  configurado; no lo cambies desde un script); o el script no conecta.
- Anota en la bitácora la versión de firmware y el número de serie del JSON.

### Peldaño 3. Medir la celda (solo la primera sesión, o si algo se movió)

```bash
uv run --extra robot python scripts/measure_cell.py --measured-by "Nombre1, Nombre2"
```

El script no mueve nada. Tú mueves el brazo con **UFactory Studio** (jog, o modo manual / guiado a
mano: el guiado a mano está permitido para **medir**, prohibido para **grabar demos**), y luego
presionas Enter. Pide, en orden: `home`, `table` (punta de la pinza tocando la mesa), `z1`…`z5`,
`tray`, los puntos `boundary_*` (extremos alcanzables, puntos justo arriba de la mesa y cerca de la
cámara de muñeca), `gripper_open` y `gripper_closed` (abre/cierra la pinza desde Studio antes de
presionar Enter). `s` salta una etiqueta; al final puedes agregar más `boundary_<nombre>` o repetir
cualquier etiqueta.

- Escribe `outputs/cell_measurements_<fecha>.yaml` (nunca en `configs/`) e imprime una caja del TCP
  marcada `PROPOSAL - review with the lab before copying`.
- **Luego, un humano** revisa el archivo con el laboratorio y copia a mano:
  - a `configs/robot/xarm6_cell.yaml`: `gripper.open_pos`, `gripper.closed_pos`, `home_deg`,
    `controller_safety.tcp_boundary_mm`; y, decididos con el laboratorio (no se miden con este
    script), `collision_sensitivity`, `reduced_max_tcp_speed_mm_s` y `gripper.speed`;
  - a `configs/scene/phase_a.yaml`: `zones_m`, `tray_m` (ya en metros) y `measured: true`.
- Haz un commit con esos cambios que diga **quién midió** y la fecha.
- **Para si:** el sondeo del peldaño 2 dio un `WARNING` de offset del TCP (las posiciones quedarían
  en otro marco), o la caja propuesta no excluye la mesa y la cámara de muñeca.

### Peldaño 4. Confirmar la seguridad del controlador en Studio

Después de la primera conexión con movimiento (peldaño 5) y en cada sesión: en UFactory Studio,
confirma que el **modo reducido**, la **frontera del TCP** y la **cerca (fence)** están activos y que
los valores son los del YAML. También puedes volver a correr el peldaño 2: el JSON trae
`reduced_states` (modo reducido, caja, velocidad, cerca).

- **Para si:** alguno aparece apagado o con valores distintos.

### Peldaño 5. Primer movimiento: 2° en una articulación

**A partir de aquí hay motores habilitados: operador del paro listo.**

```bash
uv run --extra robot python scripts/xarm_check.py --i-am-supervising --joint 6
```

Este comando **no** es de solo lectura: limpia advertencias y errores, habilita motores, configura
sensibilidad de colisión, velocidad reducida, caja del TCP, cerca, modo reducido, modo servo y la
pinza. Se niega a correr si falta algún valor medido.

- **Esperado:** la articulación 6 gira 2° y regresa, despacio; "final joints" igual al inicio.
- Repite con `--joint 5`, `4`, `3`, `2`, `1`, **una a la vez**, mirando el brazo cada vez.
- **Para si:** se mueve otra articulación, se mueve más de lo esperado, salta un error de colisión
  o el brazo no regresa.

### Peldaño 6. Cámaras

```bash
uv run python scripts/camera_check.py --probe-indices 0-5   # qué índices USB abren
uv run python scripts/camera_check.py --seconds 10
```

Antes, llena `configs/cameras/cell.yaml` con las fuentes reales (índice, `/dev/videoN` o
`rtsp://...`) y la resolución/fps acordadas (decisión abierta 10).

- **Esperado:** cada vista abre, `measured_fps` cerca del pedido, pocas `dropped_frames_est`,
  `fraction_over_max` de la desincronización ~0 y `RESULT: OK`. Revisa las fotos en
  `outputs/sessions/<fecha>/snapshot_*.png`.
- **Para si:** una vista no abre, el fps medido está muy abajo o la desincronización pasa de
  `max_skew_s` seguido.

### Peldaño 7. Lazo cerrado con política programada (sin VLA)

Primero sin mandar nada (conexión de solo lectura; registra las acciones que habría mandado):

```bash
uv run --extra robot python scripts/run_live.py --policy synthetic --system S0 --placement wrench=z3 --text "Pick up the wrench and put it in the tray" --dry-run --max-episode-s 20
```

Revisa `outputs/live_log_DRYRUN.jsonl`: entradas `would_send`, recortes (`clip`) y el resumen.

Luego con movimiento real, a **velocidad mínima**: baja antes `reduced_max_tcp_speed_mm_s` en
`configs/robot/xarm6_cell.yaml` y `max_step_deg` en `configs/safety/xarm6_minimal.yaml`; **un objeto,
una zona**:

```bash
uv run --extra robot python scripts/run_live.py --policy synthetic --system S0 --placement wrench=z3 --text "Pick up the wrench and put it in the tray" --max-episode-s 20 --i-am-supervising
```

- La política programada solo corre en el brazo real si `configs/scene/phase_a.yaml` tiene
  `measured: true` (va hacia esas coordenadas).
- **Para si:** la trayectoria del dry-run apunta a un lugar que no es la zona, hay muchos recortes,
  o el movimiento real se acerca a la mesa o a la cámara de muñeca.

### Peldaño 8. Paro por watchdog y paro de emergencia

1. Durante un movimiento del peldaño 7, presiona **Ctrl+C** en la laptop. El script sale y el
   driver manda `set_state(4)` (paro) al desconectar. Mide a ojo si el brazo se detiene dentro de
   `watchdog_timeout_s` (0.5 s en `configs/safety/xarm6_minimal.yaml`) y anótalo.
   Nota: el watchdog en software vive en el mismo proceso que la política; detecta una política
   que deja de responder, pero si el proceso muere de golpe, lo que se observa es el
   comportamiento del controlador sin nuevos objetivos. Anota lo que veas; no lo supongas.
2. Presiona **una vez a propósito** el paro de la celda con el brazo quieto o lento. Anota los pasos
   exactos para recuperar (qué se rearma, qué se hace en Studio, qué peldaño se repite).

- **Para si:** el brazo no se detiene, o la recuperación no está clara.

### Peldaño 9. Solo cuando 1–8 pasan

- Grabación por teleoperación (`scripts/record_demos.py`, ver `docs/SETUP.md` sección 4).
- Más adelante, el primer checkpoint de SmolVLA, a velocidad mínima, **solo E0 (inglés)**.

## 4. Bitácora de la sesión

Copia esta plantilla a `outputs/sessions/<fecha>/bitacora.md` y llénala durante la sesión.

```markdown
# Bitácora xArm 6 - <AAAA-MM-DD>

- Personas y roles: paro = ___, laptop = ___, registrador = ___
- Commit de git: ___ (`git rev-parse HEAD`; ¿cambios sin commit? sí/no)
- Hash de configuración: ___ (campo `config.combined_sha256` del probe JSON)
- Firmware del controlador: ___ (campo `probe.firmware` del probe JSON); SN: ___
- Archivo del sondeo: outputs/sessions/<fecha>/probe_<hora>.json

| Hora | Peldaño / comando | Resultado | Anomalías |
|---|---|---|---|
| | | | |

- Paros de emergencia presionados (hora, motivo, cómo se recuperó):
- Códigos de error / advertencia vistos (código, texto, qué se hizo):
- Pendientes para la próxima sesión:
```

## 5. Problemas comunes

- **`ping 192.168.1.226` falla:** revisa el cable y que el controlador esté encendido; que el
  adaptador Ethernet tenga la IP fija en `192.168.1.x/24`; que el Wi-Fi no esté también en
  `192.168.1.x` (`net_check.py` lo detecta); apaga VPNs.
- **Ping funciona pero los puertos 502/30002 están cerrados:** el controlador puede estar
  arrancando, o UFactory Studio u otro programa tiene la conexión. Cierra otras sesiones del SDK;
  espera a que arranque.
- **`error_code` / `warn_code` distintos de 0:** búscalos en el SDK (sin conectarte al robot):

  ```bash
  uv run --extra robot python -c "from xarm.core.config.x_code import ControllerErrorCodeMap as M; print(M.get(<codigo>, {}).get('en'))"
  ```

  (para advertencias, `ControllerWarnCodeMap`). **Límpialos en UFactory Studio, nunca con un
  script**, y anota el código y el texto en la bitácora. El sondeo no limpia nada a propósito.
- **Disparos de colisión:** anota qué se estaba haciendo; revisa en Studio; no subas la
  sensibilidad para "que ya no pase" sin decidirlo con el laboratorio.
- **La FK del URDF y la pose del controlador no coinciden:** casi siempre es el offset del TCP
  (`WARNING ... NOT comparable` en el peldaño 2). Confirma la herramienta montada y el offset en
  Studio. Mientras no coincidan, no copies mediciones de zonas.
- **Una cámara no abre:** índice equivocado (usa `--probe-indices`), otra aplicación la tiene
  abierta (Zoom, Teams, Studio, el navegador), o el cable USB/hub no da suficiente corriente.

## 6. Apagado

1. Lleva el brazo a `home` a baja velocidad (desde UFactory Studio).
2. Deshabilita el movimiento (motores) en Studio.
3. Cierra los scripts (se desconectan solos) y Studio.
4. Apaga según el orden que indique el laboratorio.
5. Termina la bitácora y guárdala en `outputs/sessions/<fecha>/`.

## 7. Recordatorios

- Las cifras de LIBERO son de **simulación**; dilo siempre.
- Ningún ensayo con condiciones en español antes de **congelar el preregistro**
  (`scripts/freeze_prereg.py --freeze`).
- **Nunca** se descarta un ensayo por su resultado. Solo se repiten fallas de hardware documentadas
  **antes** de ver la decisión.
