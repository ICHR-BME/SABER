# Checklist de seguridad para sesiones con el xArm 6

Corto a propósito. La celda aporta la seguridad perimetral (paro de emergencia y protección de
acceso). SABER agrega solo lo mínimo para que una acción equivocada del VLA no choque el brazo
(PROPUESTA 2.2.5). **Nada de esto está certificado** (ni ISO 10218 ni ISO/TS 15066). No describas
el sistema como seguro para colaboración.

## Antes de la primera sesión en la celda (una vez)

- [ ] Conocer el procedimiento propio de la celda: ubicación del paro, reinicio, reglas de acceso.
- [ ] Llenar cada `null` de `configs/robot/xarm6_cell.yaml` con valores **medidos**: IP del
      controlador, posiciones abierta/cerrada de la pinza, sensibilidad de colisión, velocidad TCP
      reducida y caja del TCP `[x_max, x_min, y_max, y_min, z_max, z_min]` (mm, marco de la base)
      que mantenga el TCP sobre la mesa y la cámara de muñeca sin choques. El driver no se conecta
      mientras alguno siga en `null`.
- [ ] Si hace falta, estrechar los rangos articulares en `configs/safety/xarm6_minimal.yaml` (solo
      pueden ser más estrechos que los del URDF oficial).
- [ ] Reemplazar las coordenadas sintéticas de `configs/scene/phase_a.yaml` y poner `measured: true`.

## Cada sesión

- [ ] Dos personas en cualquier ejecución autónoma. Una sostiene el paro de la celda todo el tiempo.
- [ ] Nadie dentro del espacio de trabajo durante un ensayo.
- [ ] Revisión de solo lectura: `uv run --extra robot python scripts/xarm_check.py` (articulaciones,
      TCP por FK del URDF y por FK del controlador; deben coincidir en pocos mm).
- [ ] Revisión de movimiento: `uv run --extra robot python scripts/xarm_check.py --i-am-supervising --joint 6`
      (mueve la articulación 6 dos grados y regresa).
- [ ] Al conectar, el driver configura: sensibilidad de detección de colisión, velocidad TCP máxima
      reducida, caja del TCP + modo cerca (fence), modo reducido activo y modo servo. Confirma en
      xArm Studio que el modo reducido y la frontera están activos.
- [ ] Límites en software en cada acción: recorte articular, salto máximo por paso, recorte de la
      pinza y paro por watchdog si la política deja de responder (`watchdog_timeout_s`).
- [ ] **La primera ejecución de cada política (o checkpoint) nueva, a velocidad mínima**: baja
      `reduced_max_tcp_speed_mm_s` y `max_step_deg`, corre E0 una vez y observa todo el movimiento.
- [ ] Nada de cautín encendido en la celda (el carrete de estaño es una réplica).

## Si algo sale mal

1. Presiona el paro de la celda.
2. Anota lo que pasó antes de tocar nada (id del ensayo, hora, qué hizo el brazo).
3. Un ensayo con falla de hardware documentada **antes** de la decisión se repite y se reporta;
   nunca se descarta por su resultado (regla de exclusión del preregistro).
