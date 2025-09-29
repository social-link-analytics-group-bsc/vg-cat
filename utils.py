import os
import yaml
import shutil
import logging
from pathlib import Path

def setup_logging(log_path=None, level=logging.INFO):
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def print_title(text, level=1, style='auto', width=None, emoji=None,
                padding=2, pad_lines=0, max_width=100, min_width=20,
                border_char=None):
    """
    Imprime títulos con 3 niveles:
      level=1 -> banner de 3 líneas con '='
      level=2 -> banner de 3 líneas con '-'
      level=3 -> línea única: guiones a ambos lados, texto centrado
    style: 'auto'|'compact'|'fixed'|'full'
    border_char: carácter para el borde (por defecto '=' para level1, '-' para otros)
    """
    txt = f"{emoji} {text}" if emoji else text
    # detectar ancho del terminal
    try:
        term_w = shutil.get_terminal_size().columns
    except Exception:
        term_w = 80

    # decidir ancho
    if style == 'full':
        w = term_w
    elif style == 'fixed':
        w = int(width) if width is not None else 80
    elif style == 'compact':
        w = len(txt) + 2 * padding
    else:  # auto
        desired = len(txt) + 2 * padding
        w = min(max(desired, min_width), min(max_width, term_w))

    w = max(min_width, int(w))

    # elegir carácter borde
    if border_char is None:
        if level == 1:
            border_char = '='
        else:
            border_char = '-'

    # construir salida
    if level == 3:
        # texto centrado con relleno a ambos lados
        txt_with_spaces = f" {txt} "
        if len(txt_with_spaces) >= w:
            out = txt  # no cabe, imprimir tal cual
        else:
            # calcular cuánto relleno a cada lado
            total_fill = w - len(txt_with_spaces)
            left_fill = total_fill // 2
            right_fill = total_fill - left_fill
            out = f"{border_char * left_fill}{txt_with_spaces}{border_char * right_fill}"
    else:
        middle = txt.center(w)
        border = border_char * w
        out = f"{border}\n{middle}\n{border}"

    # imprimir con pad_lines
    for _ in range(pad_lines):
        print()
    print(out)
    for _ in range(pad_lines):
        print()