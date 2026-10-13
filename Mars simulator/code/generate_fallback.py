#!/usr/bin/env python3
"""
Генератор резервных эфемерид из JPL Horizons.

Запрашивает векторы состояния Земли (399) и Марса (499)
в гелиоцентрической системе (CENTER='500@10'), эклиптика J2000.
Сохраняет результат в формате JavaScript для horizons-fallback.js
и в JSON для дальнейшего использования.

Использование:
    pip install requests
    python generate_fallback.py
"""

import argparse
import json
import sys
from pathlib import Path

import requests

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
EARTH_ID = "399"
MARS_ID = "499"


def fetch_vectors(body_id: str, start: str, stop: str, step: str = "10d") -> str:
    """Запрашивает у Horizons таблицу векторов и возвращает текстовый ответ."""
    params = {
        "format": "json",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",         # Гелиоцентр
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "'2'",            # Позиция + скорость
        "OUT_UNITS": "'KM-S'",
        "VEC_LABELS": "'NO'",
        "CSV_FORMAT": "'YES'",
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "result" not in data:
        raise RuntimeError(f"Horizons вернул неожиданный ответ: {data}")
    return data["result"]


def parse_vectors(text: str) -> list[list[float]]:
    """
    Парсит таблицу между $$SOE и $$EOE.
    Формат строки: JD, Calendar, X, Y, Z, VX, VY, VZ
    (в CSV-формате могут быть запятые).
    """
    rows: list[list[float]] = []
    inside = False
    for line in text.splitlines():
        if "$$SOE" in line:
            inside = True
            continue
        if "$$EOE" in line:
            break
        if not inside:
            continue

        # CSV-формат: значения через запятую
        line = line.strip()
        if not line:
            continue

        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()

        if len(parts) < 8:
            continue

        try:
            # В CSV-формате поля: JD, Cal, X, Y, Z, VX, VY, VZ
            jd = float(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
        except (ValueError, IndexError):
            continue

        rows.append([jd, x, y, z, vx, vy, vz])

    if not rows:
        raise RuntimeError("Не удалось извлечь ни одной строки из ответа Horizons")
    return rows


def to_js_array(name: str, rows: list[list[float]], indent: str = "  ") -> str:
    """Преобразует список векторов в JavaScript-массив."""
    lines = [f"const {name} = ["]
    for r in rows:
        formatted = ", ".join(f"{v:.8e}" for v in r)
        lines.append(f"{indent}[{formatted}],")
    lines.append("];")
    return "\n".join(lines)


def to_js_fallback(earth: list[list[float]], mars: list[list[float]]) -> str:
    """Собирает полный JS-файл horizons-fallback.js."""
    header = (
        "/**\n"
        " * Резервные эфемериды из JPL Horizons.\n"
        " * Формат: [JD_TDB, x, y, z, vx, vy, vz] — гелиоцентрическая эклиптика J2000.\n"
        " * Единицы: км, км/с.\n"
        " * Источник: https://ssd.jpl.nasa.gov/api/horizons.api\n"
        " * COMMAND='399' (Earth), '499' (Mars), CENTER='500@10' (Sun),\n"
        " * EPHEM_TYPE='VECTORS', REF_PLANE='ECLIPTIC'.\n"
        " * Сгенерировано автоматически: generate_fallback.py\n"
        " */\n"
    )
    body = f"""
const FALLBACK_EPHEM = (function(){{
{to_js_array("EARTH", earth)}

{to_js_array("MARS", mars)}

  function lerp(a, b, t) {{ return a + (b - a) * t; }}

  function interp(table, jd) {{
    if (jd <= table[0][0]) return vec(table[0]);
    const last = table[table.length - 1];
    if (jd >= last[0]) return vec(last);

    let lo = 0, hi = table.length - 1;
    while (hi - lo > 1) {{
      const mid = (lo + hi) >> 1;
      if (table[mid][0] <= jd) lo = mid; else hi = mid;
    }}
    const A = table[lo], B = table[hi];
    const t = (jd - A[0]) / (B[0] - A[0]);
    return {{
      x:  lerp(A[1], B[1], t),
      y:  lerp(A[2], B[2], t),
      z:  lerp(A[3], B[3], t),
      vx: lerp(A[4], B[4], t),
      vy: lerp(A[5], B[5], t),
      vz: lerp(A[6], B[6], t),
    }};
  }}

  function vec(r) {{
    return {{ x: r[1], y: r[2], z: r[3], vx: r[4], vy: r[5], vz: r[6] }};
  }}

  return {{
    earth: jd => interp(EARTH, jd),
    mars:  jd => interp(MARS,  jd),
    available: true,
    source: 'fallback',
  }};
}})();
"""
    return header + body


def main():
    parser = argparse.ArgumentParser(description="Генерация эфемерид для симулятора")
    parser.add_argument("--start", default="2026-01-01", help="Начало диапазона")
    parser.add_argument("--stop",  default="2036-01-01", help="Конец диапазона")
    parser.add_argument("--step",  default="10d", help="Шаг (например 10d, 1d, 30d)")
    parser.add_argument("--out",   default="horizons-fallback.js", help="Выходной JS")
    parser.add_argument("--json",  default="ephemeris.json", help="Выходной JSON")
    args = parser.parse_args()

    print(f"Запрос Земли ({EARTH_ID}) с {args.start} по {args.stop}, шаг {args.step}…")
    earth_text = fetch_vectors(EARTH_ID, args.start, args.stop, args.step)
    earth_rows = parse_vectors(earth_text)
    print(f"  Получено {len(earth_rows)} точек")

    print(f"Запрос Марса ({MARS_ID}) с {args.start} по {args.stop}, шаг {args.step}…")
    mars_text = fetch_vectors(MARS_ID, args.start, args.stop, args.step)
    mars_rows = parse_vectors(mars_text)
    print(f"  Получено {len(mars_rows)} точек")

    js = to_js_fallback(earth_rows, mars_rows)
    Path(args.out).write_text(js, encoding="utf-8")
    print(f"Записано: {args.out}")

    data = {"earth": earth_rows, "mars": mars_rows,
            "start": args.start, "stop": args.stop, "step": args.step}
    Path(args.json).write_text(json.dumps(data), encoding="utf-8")
    print(f"Записано: {args.json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)