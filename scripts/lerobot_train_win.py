"""Run ``lerobot-train`` (lerobot 0.4.4) on Windows without symlink privileges.

lerobot points ``checkpoints/last`` to the newest checkpoint with a symlink, which Windows refuses
without Developer Mode or admin rights (WinError 1314). Training then crashes right after saving the
first checkpoint (seen 2026-11-19 at step 5000). This wrapper keeps the symlink when it can and
falls back to a directory junction (``mklink /J``, no privileges needed) otherwise. Arguments are
passed to lerobot-train unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    last = checkpoint_dir.parent / "last"
    if os.path.lexists(last):
        if last.is_symlink():
            last.unlink()
        else:
            os.rmdir(last)  # removes a junction without touching its target
    try:
        last.symlink_to(checkpoint_dir.relative_to(checkpoint_dir.parent), target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(last), str(checkpoint_dir.resolve())],
            check=True,
            capture_output=True,
        )
    return last


def main() -> int:
    import lerobot.scripts.lerobot_train as lerobot_train
    import lerobot.utils.train_utils as train_utils

    train_utils.update_last_checkpoint = update_last_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint
    sys.argv[0] = "lerobot-train"
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
