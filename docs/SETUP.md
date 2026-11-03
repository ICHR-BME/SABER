# Instalación y puesta en marcha

Funciona en Windows 11 (laptop con RTX 5060) y en Linux (CI, GPU rentada). Todos los comandos se
corren desde la raíz del repositorio.

## 1. Desde cero

1. Python 3.10–3.12 (desarrollo con 3.11).
2. Instalar `uv`:
   - Linux / macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`, o
     `python -m pip install --user uv` (queda en `%APPDATA%\Python\Python311\Scripts`; agrégalo al PATH).
3. Clonar e instalar lo básico (sin GPU, suficiente para todas las pruebas):

```bash
git clone https://github.com/ICHR-BME/SABER.git
cd SABER
uv sync
uv run pytest
```

4. Extras según lo que vayas a hacer:

| Extra | Para qué | Comando |
|---|---|---|
| `robot` | xArm 6 real (xarm-python-sdk 1.18.4) | `uv sync --extra robot` |
| `teleop` | GELLO (Dynamixel), SpaceMouse, teclado | `uv sync --extra teleop` |
| `asr` | Whisper (faster-whisper 1.2.1) | `uv sync --extra asr` |
| `lerobot` | LeRobot 0.4.4 + SmolVLA/ACT, LeRobotDataset | `uv sync --extra lerobot` |
| `interp` | intérprete local de Hugging Face | `uv sync --extra interp` |
| `sim` | LIBERO (solo Linux) | `uv sync --extra sim` |

Se pueden combinar: `uv sync --extra robot --extra lerobot --extra asr --extra interp`.
PyTorch se instala desde el índice **cu128** de PyTorch (necesario para la RTX 5060). Verifica:

```bash
uv run --extra lerobot python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Conexión al xArm 6

Lee `docs/SAFETY.md` antes. Nunca solo; paro de la celda en la mano.

1. Conecta la laptop a la red del controlador por Ethernet y fija una IP en la misma subred.
2. Llena en `configs/robot/xarm6_cell.yaml` **con valores medidos**: `robot.ip`, posiciones de la
   pinza (`open_pos`, `closed_pos`), `controller_safety.collision_sensitivity`,
   `reduced_max_tcp_speed_mm_s` y `tcp_boundary_mm`. Mientras alguno sea `null`, el driver no
   se conecta.
3. Prueba de solo lectura y luego de movimiento (2° en una articulación):

```bash
uv run --extra robot python scripts/xarm_check.py
uv run --extra robot python scripts/xarm_check.py --i-am-supervising --joint 6
```

   Compara "TCP (URDF FK)" con "TCP (controller FK)": deben coincidir en pocos milímetros con la
   pinza montada. Si no, revisa `tcp_offset_m`.
4. Sin robot puedes probar todo con `--mock`.

## 3. Cámaras de la celda y de muñeca

1. Edita `configs/cameras/cell.yaml`: una entrada por vista (`wrist` y las fijas de la celda). `source`
   puede ser un índice (USB; DirectShow en Windows, V4L2 en Linux), una ruta `/dev/videoN` o un
   URL `rtsp://...` para cámaras IP.
2. Para encontrar índices puedes usar la herramienta de LeRobot (`uv run --extra lerobot lerobot-find-cameras --help`)
   o probar índices 0, 1, 2… con `configs/cameras/cell.yaml`.
3. Las vistas se leen juntas y se rechazan si sus marcas de tiempo difieren más que `max_skew_s`.

## 4. Teleoperación y grabación de demos

- **GELLO** (preferido): llena `configs/teleop/gello_xarm6.yaml` (puerto, ids, signos), pon el GELLO
  y el brazo en la misma pose y calibra:

```bash
uv run --extra robot --extra teleop python scripts/record_demos.py --calibrate-gello --i-am-supervising
```

- **SpaceMouse**: `--teleop spacemouse` (botón 0 abre/cierra la pinza, botón 1 termina el episodio).
- **Teclado**: solo depuración (`--teleop keyboard`).

Antes de grabar, mide los centros de las cinco zonas y la charola en `configs/scene/phase_a.yaml`
y pon `measured: true`. Grabación guiada (el script dice qué objetivo y qué zona toca):

```bash
uv run --extra robot --extra teleop python scripts/record_demos.py --teleop gello --operator ian \
    --n-per-goal 50 --out data/datasets/t1 --i-am-supervising
```

Sin robot: `uv run python scripts/record_demos.py --mock --teleop synthetic --n-per-goal 2 --out outputs/demos --yes`.

## 5. Ajuste fino (GPU rentada)

```bash
uv run python scripts/train_policy.py configs/train/smolvla_xarm6_en.yaml          # ver comando y estimación
uv run --extra lerobot python scripts/train_policy.py configs/train/smolvla_xarm6_en.yaml --run
```

Convierte primero el dataset npz a LeRobotDataset con `saber.record.writer.npz_to_lerobot` (o graba
directo con `--writer lerobot`).

## 6. Evaluación

```bash
uv run python scripts/make_trials.py --out outputs/trials --seed <semilla>       # lista de ensayos + hojas
uv run python scripts/build_odpb.py --dataset data/datasets/t1 --policy smolvla:<ruta> --interpreter hf:<modelo> --out outputs/odpb.csv
uv run python scripts/calibrate_gate.py outputs/odpb.csv
uv run python scripts/report.py --trials <resultados.csv> --odpb outputs/odpb.csv --out outputs/report
bash scripts/gpu_libero.sh                                                       # LIBERO-Goal, Linux + GPU
```
