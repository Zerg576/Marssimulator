#!/usr/bin/env python3
"""
Интерактивный симулятор межпланетных запусков Земля → Марс.
Версия для запуска из исходников и для сборки в .exe (PyInstaller).

Использование:
    python mars_simulator.py        # запуск из исходников
    mars_simulator.exe              # запуск собранного приложения
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider


# ============================================================
# ПУТИ: КУДА СОХРАНЯТЬ КЭШ
# ============================================================
def app_dir() -> Path:
    """Папка, где лежит .exe (или .py при запуске из исходников)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


APP_DIR = app_dir()
CACHE_PATH = APP_DIR / "horizons_cache.json"


# ============================================================
# КОНСТАНТЫ
# ============================================================
MU_SUN   = 1.32712440018e11   # км³/с²
AU       = 149597870.7        # км
MU_EARTH = 398600.4418        # км³/с²
MU_MARS  = 42828.37           # км³/с²
R_EARTH  = 6378.137           # км
R_MARS   = 3396.19            # км

TWO_PI   = 2.0 * math.pi
DAY_SEC  = 86400.0
J2000_JD = 2451545.0

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


# ============================================================
# КЛИЕНТ JPL HORIZONS
# ============================================================
class HorizonsClient:
    """Клиент JPL Horizons с кэшем на диске (рядом с .exe / .py)."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: dict[str, list[list[float]]] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        except Exception as e:
            print(f"[Horizons] Не удалось сохранить кэш: {e}")

    def fetch(self, body_id: str, start: str, stop: str,
              step: str = "1d") -> list[list[float]]:
        key = f"{body_id}_{start}_{stop}_{step}"
        if key in self.cache and len(self.cache[key]) > 5:
            print(f"[Horizons] Из кэша: {body_id}, {len(self.cache[key])} точек")
            return self.cache[key]

        print(f"[Horizons] Запрос {body_id}: {start} … {stop}, шаг {step}")
        params = {
            "format": "json",
            "COMMAND": f"'{body_id}'",
            "OBJ_DATA": "'NO'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'VECTORS'",
            "CENTER": "'500@10'",
            "REF_PLANE": "'ECLIPTIC'",
            "REF_SYSTEM": "'ICRF'",
            "START_TIME": f"'{start}'",
            "STOP_TIME": f"'{stop}'",
            "STEP_SIZE": f"'{step}'",
            "VEC_TABLE": "'2'",
            "OUT_UNITS": "'KM-S'",
            "VEC_LABELS": "'NO'",
            "CSV_FORMAT": "'YES'",
        }
        r = requests.get(HORIZONS_URL, params=params, timeout=60)
        r.raise_for_status()
        text = r.json().get("result", "")
        rows = self._parse(text)
        if not rows:
            raise RuntimeError(f"Horizons не вернул данные для {body_id}")
        self.cache[key] = rows
        self._save_cache()
        return rows

    @staticmethod
    def _parse(text: str) -> list[list[float]]:
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
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            if len(parts) < 8:
                continue
            try:
                jd = float(parts[0])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
            except (ValueError, IndexError):
                continue
            rows.append([jd, x, y, z, vx, vy, vz])
        return rows


# ============================================================
# ЭФЕМЕРИДЫ
# ============================================================
@dataclass
class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


class Ephemeris:
    """Интерполяция табличных эфемерид Horizons."""

    def __init__(self, earth_rows, mars_rows):
        self.earth = np.array(earth_rows, dtype=float)
        self.mars  = np.array(mars_rows,  dtype=float)

    @staticmethod
    def _interp(table: np.ndarray, jd: float) -> State:
        jds = table[:, 0]
        if jd <= jds[0]:
            row = table[0]
        elif jd >= jds[-1]:
            row = table[-1]
        else:
            idx = np.searchsorted(jds, jd)
            a, b = table[idx - 1], table[idx]
            t = (jd - a[0]) / (b[0] - a[0])
            row = a + (b - a) * t
        return State(row[1], row[2], row[3], row[4], row[5], row[6])

    def earth_at(self, jd: float) -> State:
        return self._interp(self.earth, jd)

    def mars_at(self, jd: float) -> State:
        return self._interp(self.mars, jd)


# ============================================================
# ЗАДАЧА ЛАМБЕРТА
# ============================================================
def lambert(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float,
            prograde: bool = True):
    """Возвращает (v1, v2) для перелёта r1 → r2 за dt секунд."""
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    cosd = float(np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0))
    dnu = math.acos(cosd)
    cross_z = r1[0] * r2[1] - r1[1] * r2[0]
    if prograde and cross_z < 0:
        dnu = TWO_PI - dnu
    elif not prograde and cross_z > 0:
        dnu = TWO_PI - dnu
    if abs(dnu - math.pi) < 1e-6:
        dnu += 1e-6

    A = math.sin(dnu) * math.sqrt(r1n * r2n / (1 - math.cos(dnu)))
    if not math.isfinite(A) or abs(A) < 1e-12:
        return None

    psi = 0.0
    lo, hi = -4 * math.pi ** 2, 4 * math.pi ** 2
    c2, c3 = 0.5, 1 / 6
    y = chi = dtc = 0.0

    for _ in range(60):
        if psi > 1e-6:
            sp = math.sqrt(psi)
            c2 = (1 - math.cos(sp)) / psi
            c3 = (sp - math.sin(sp)) / (psi * sp)
        elif psi < -1e-6:
            sp = math.sqrt(-psi)
            c2 = (1 - math.cosh(sp)) / psi
            c3 = (math.sinh(sp) - sp) / (-psi) ** 1.5
        else:
            c2, c3 = 0.5, 1 / 6

        y = r1n + r2n + A * (psi * c3 - 1) / math.sqrt(c2)

        if A > 0 and y < 0:
            k = 0
            while y < 0 and k < 400:
                psi += 0.05
                k += 1
                if psi > 1e-6:
                    sp = math.sqrt(psi)
                    c2 = (1 - math.cos(sp)) / psi
                    c3 = (sp - math.sin(sp)) / (psi * sp)
                else:
                    c2, c3 = 0.5, 1 / 6
                y = r1n + r2n + A * (psi * c3 - 1) / math.sqrt(c2)

        chi = math.sqrt(y / c2)
        dtc = (chi ** 3 * c3 + A * math.sqrt(y)) / math.sqrt(mu)

        if dtc <= dt:
            lo = psi
        else:
            hi = psi

        prev = psi
        psi = 0.5 * (lo + hi)
        if abs(dtc - dt) < 1e-7 * dt or abs(psi - prev) < 1e-13:
            break

    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    if abs(g) < 1e-12:
        return None
    gd = 1 - y / r2n

    v1 = (r2 - f * r1) / g
    v2 = (gd * r2 - r1) / g
    return v1, v2


# ============================================================
# ИМПУЛЬСЫ
# ============================================================
def dv_from_parking(v_inf: float, r_park: float, mu_planet: float) -> float:
    v_park = math.sqrt(mu_planet / r_park)
    return math.sqrt(v_inf ** 2 + 2 * mu_planet / r_park) - v_park


# ============================================================
# РЕЗУЛЬТАТ ДНЯ
# ============================================================
@dataclass
class DayResult:
    jd: float
    tof: float
    dv: float
    dv_dep: float
    dv_arr: float
    v_inf_dep: float
    v_inf_arr: float
    c3: float
    r1: np.ndarray
    v1: np.ndarray
    r2: np.ndarray
    v2: np.ndarray
    earth_dep: State
    mars_dep: State
    mars_arr: State
    success: bool
    reasons: dict = field(default_factory=dict)
    score: float = 0.0


def evaluate_day(eph: Ephemeris, jd: float, opts: dict):
    earth_dep = eph.earth_at(jd)
    mars_dep  = eph.mars_at(jd)
    r1 = earth_dep.pos
    best = None

    for tof in opts["tof_list"]:
        arr_jd = jd + tof
        mars_arr = eph.mars_at(arr_jd)
        r2 = mars_arr.pos
        sol = lambert(r1, r2, tof * DAY_SEC, MU_SUN, True)
        if sol is None:
            continue
        v1, v2 = sol

        v_inf_dep = float(np.linalg.norm(v1 - earth_dep.vel))
        v_inf_arr = float(np.linalg.norm(mars_arr.vel - v2))
        c3 = v_inf_dep ** 2

        dv_dep = dv_from_parking(v_inf_dep, opts["r_park_e"], MU_EARTH)
        dv_arr = dv_from_parking(v_inf_arr, opts["r_park_m"], MU_MARS)
        dv = dv_dep + dv_arr

        if best is None or dv < best.dv:
            best = DayResult(
                jd=jd, tof=tof, dv=dv,
                dv_dep=dv_dep, dv_arr=dv_arr,
                v_inf_dep=v_inf_dep, v_inf_arr=v_inf_arr, c3=c3,
                r1=r1, v1=v1, r2=r2, v2=v2,
                earth_dep=earth_dep, mars_dep=mars_dep, mars_arr=mars_arr,
                success=False,
            )

    if best is None:
        return None

    reasons = {
        "dv":  best.dv <= opts["dv_budget"],
        "tof": best.tof <= opts["tof_max"],
    }
    best.reasons = reasons
    best.success = all(reasons.values())
    best.score = max(0.0, min(1.0,
        (opts["dv_budget"] - best.dv) / (0.35 * opts["dv_budget"])))
    return best


# ============================================================
# ПРИЛОЖЕНИЕ
# ============================================================
class MarsSimulator:
    def __init__(self, eph: Ephemeris, start_year: int = 2026,
                 end_year: int = 2035):
        self.eph = eph
        self.start_year = start_year
        self.end_year = end_year
        self.results: dict[float, DayResult] = {}
        self.selected_jd = None

        self.dv_budget = 6.0
        self.r_park_e = R_EARTH + 200
        self.r_park_m = R_MARS + 400
        self.tof_max = 350.0
        self.tof_mode = "auto"
        self.tof_fixed = 250.0

        # Главное окно
        self.fig = plt.figure(figsize=(15, 8.5), num="Марс · симулятор")
        self.fig.canvas.manager.set_window_title(
            "Симулятор межпланетных запусков · Графика")
        gs = GridSpec(2, 3, width_ratios=[2.6, 0.9, 0.9],
                      height_ratios=[2.2, 1.0],
                      hspace=0.30, wspace=0.25, figure=self.fig)

        self.ax_sky = self.fig.add_subplot(gs[0, 0])
        self.ax_cal = self.fig.add_subplot(gs[0, 1:])
        self.ax_info = self.fig.add_subplot(gs[1, :])

        self._setup_sky()
        self._setup_calendar()
        self._setup_info_panel()
        self._add_sliders()
        self._add_buttons()

        self.info_window = self.ax_info.figure
        self.info_window.canvas.mpl_connect("button_press_event",
                                            self._on_info_click)

        self._compute_all()
        self._pick_best()
        self._refresh()

    # ---------- сцена ----------
    def _setup_sky(self):
        ax = self.ax_sky
        ax.set_aspect("equal")
        ax.set_facecolor("#060a0f")
        ax.set_xlim(-2.0 * AU, 2.0 * AU)
        ax.set_ylim(-2.0 * AU, 2.0 * AU)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#22303d")

        ax.add_patch(Circle((0, 0), 1.0 * AU, fill=False,
                            color="#5a96c8", lw=0.7, alpha=0.35))
        ax.add_patch(Circle((0, 0), 1.524 * AU, fill=False,
                            color="#dc785a", lw=0.7, alpha=0.35))
        ax.add_patch(Circle((0, 0), 0.05 * AU, color="#ffe9a8", zorder=5))
        ax.add_patch(Circle((0, 0), 0.12 * AU, color="#ffb040",
                            alpha=0.35, zorder=4))

        self.earth_dot, = ax.plot([], [], "o", color="#4fa8ff", ms=9, zorder=6)
        self.mars_dot,  = ax.plot([], [], "o", color="#e05555", ms=8, zorder=6)
        self.mars_arr_dot, = ax.plot([], [], "o", color="#e05555", ms=8,
                                     mfc="none", mec="#e05555",
                                     mew=1.6, zorder=6)
        self.traj_line, = ax.plot([], [], "--", color="#ffd24a",
                                  lw=1.6, zorder=5)
        self.sat_dot,  = ax.plot([], [], "o", color="#fffbe6", ms=7, zorder=7)

    # ---------- календарь ----------
    def _setup_calendar(self):
        ax = self.ax_cal
        ax.set_facecolor("#0b1117")
        ax.set_title("Календарь окон запуска", color="#8b98a8",
                     fontsize=10, loc="left")
        for spine in ax.spines.values():
            spine.set_color("#22303d")

        n_years = self.end_year - self.start_year + 1
        self.cal_grid = np.full((n_years, 366), -1.0)
        self.cal_im = ax.imshow(
            self.cal_grid, aspect="auto", interpolation="nearest",
            cmap=self._make_cmap(), vmin=-1.0, vmax=1.0,
            extent=[0, 366, n_years, 0],
        )
        ax.set_yticks(np.arange(n_years) + 0.5)
        ax.set_yticklabels([str(self.start_year + i)
                            for i in range(n_years)],
                           color="#8b98a8", fontsize=8)
        ax.set_xticks([1, 32, 60, 91, 121, 152,
                       182, 213, 244, 274, 305, 335])
        ax.set_xticklabels(["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                            "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"],
                           color="#8b98a8", fontsize=8)
        ax.tick_params(colors="#8b98a8", length=0)

        self.fig.canvas.mpl_connect("button_press_event",
                                    self._on_calendar_click)

    @staticmethod
    def _make_cmap():
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            "cal", ["#161f29", "#3c1f1f", "#e05555", "#f0b429", "#3cdc82"])

    # ---------- окно с таблицей ----------
    def _setup_info_panel(self):
        self.fig_info = plt.figure(figsize=(8.5, 7.0),
                                   num="Марс · численный итог")
        self.fig_info.canvas.manager.set_window_title(
            "Численный итог · Земля → Марс")
        self.fig_info.patch.set_facecolor("#121b24")

        self.ax_info = self.fig_info.add_subplot(111)
        self.ax_info.set_facecolor("#121b24")
        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        self.ax_info.set_xlim(0, 1); self.ax_info.set_ylim(0, 1)
        for spine in self.ax_info.spines.values():
            spine.set_color("#22303d")

        self.ax_info.text(
            0.5, 0.965, "ЧИСЛЕННЫЙ ИТОГ МОДЕЛИРОВАНИЯ",
            transform=self.ax_info.transAxes,
            color="#8b98a8", fontsize=12, ha="center", va="top",
            family="monospace", weight="bold")

        self.ax_info.plot([0.05, 0.95], [0.925, 0.925],
                          color="#2b3a4a", lw=1.0,
                          transform=self.ax_info.transAxes)

        self._info_artists: list = []

        self.verdict_text = self.ax_info.text(
            0.5, 0.045, "— ОЖИДАНИЕ —",
            transform=self.ax_info.transAxes,
            color="#8b98a8", fontsize=13, ha="center", va="center",
            family="monospace", weight="bold")

        self.ax_info.plot([0.05, 0.95], [0.095, 0.095],
                          color="#2b3a4a", lw=1.0,
                          transform=self.ax_info.transAxes)

        ax_btn = self.fig_info.add_axes([0.30, 0.905, 0.40, 0.055])
        ax_btn.set_facecolor("#16212c")
        self.b_raise = Button(ax_btn, "◀ ПЕРЕЙТИ К ГРАФИКЕ",
                              color="#16212c", hovercolor="#22303d")
        self.b_raise.label.set_color("#4fc3f7")
        self.b_raise.on_clicked(lambda _: self._raise_main_window())

    def _raise_main_window(self):
        try:
            mng = self.fig.canvas.manager
            if hasattr(mng, "window"):
                mng.window.raise_()
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _raise_info_window(self):
        try:
            mng = self.info_window.canvas.manager
            if hasattr(mng, "window"):
                mng.window.raise_()
            self.info_window.canvas.draw_idle()
        except Exception:
            pass

    def _on_info_click(self, event):
        if event.inaxes is self.ax_info:
            self._refresh()

    # ---------- слайдеры ----------
    def _add_sliders(self):
        self.fig.subplots_adjust(bottom=0.22, top=0.95,
                                 left=0.05, right=0.98)

        ax_dv = self.fig.add_axes([0.10, 0.14, 0.28, 0.025])
        ax_he = self.fig.add_axes([0.10, 0.10, 0.28, 0.025])
        ax_hm = self.fig.add_axes([0.10, 0.06, 0.28, 0.025])
        ax_tf = self.fig.add_axes([0.55, 0.14, 0.28, 0.025])
        ax_fx = self.fig.add_axes([0.55, 0.10, 0.28, 0.025])

        self.s_dv = Slider(ax_dv, "Δv budget, км/с", 4.5, 12.0,
                           valinit=self.dv_budget, valstep=0.1,
                           color="#4fc3f7")
        self.s_he = Slider(ax_he, "h парк. Земли, км", 150, 1200,
                           valinit=self.r_park_e - R_EARTH,
                           valstep=10, color="#4fc3f7")
        self.s_hm = Slider(ax_hm, "h парк. Марса, км", 150, 2000,
                           valinit=self.r_park_m - R_MARS,
                           valstep=10, color="#4fc3f7")
        self.s_tf = Slider(ax_tf, "TOF max, сут", 180, 500,
                           valinit=self.tof_max, valstep=5,
                           color="#f0b429")
        self.s_fx = Slider(ax_fx, "Фиксир. TOF, сут", 120, 450,
                           valinit=self.tof_fixed, valstep=5,
                           color="#f0b429")

        for s, c in [(self.s_dv, "#4fc3f7"), (self.s_he, "#4fc3f7"),
                     (self.s_hm, "#4fc3f7"), (self.s_tf, "#f0b429"),
                     (self.s_fx, "#f0b429")]:
            s.label.set_color("#8b98a8")
            s.valtext.set_color(c)
            s.on_changed(self._on_slider)

    def _on_slider(self, _=None):
        self.dv_budget = float(self.s_dv.val)
        self.r_park_e = R_EARTH + float(self.s_he.val)
        self.r_park_m = R_MARS  + float(self.s_hm.val)
        self.tof_max  = float(self.s_tf.val)
        self.tof_fixed = float(self.s_fx.val)

    # ---------- кнопки ----------
    def _add_buttons(self):
        ax_run  = self.fig.add_axes([0.08, 0.005, 0.17, 0.035])
        ax_best = self.fig.add_axes([0.27, 0.005, 0.17, 0.035])
        ax_info = self.fig.add_axes([0.46, 0.005, 0.17, 0.035])

        self.b_run  = Button(ax_run,  "ПЕРЕСЧИТАТЬ",
                             color="#4fc3f7", hovercolor="#7ad0ff")
        self.b_best = Button(ax_best, "ЛУЧШЕЕ ОКНО",
                             color="#16212c", hovercolor="#22303d")
        self.b_info = Button(ax_info, "ОКНО С ЧИСЛАМИ",
                             color="#16212c", hovercolor="#22303d")

        self.b_run.label.set_color("#04121b")
        self.b_best.label.set_color("#d8e2ec")
        self.b_info.label.set_color("#4fc3f7")

        self.b_run.on_clicked(
            lambda _: (self._compute_all(), self._pick_best(), self._refresh()))
        self.b_best.on_clicked(
            lambda _: (self._pick_best(), self._refresh()))
        self.b_info.on_clicked(lambda _: self._raise_info_window())

    # ---------- расчёт ----------
    def _tof_list(self):
        if self.tof_mode == "fixed":
            return [self.tof_fixed]
        return [float(t) for t in range(120, 421, 20)]

    def _compute_all(self):
        print("[Sim] Пересчёт всех дней…")
        self.results.clear()
        opts = {
            "dv_budget": self.dv_budget,
            "r_park_e": self.r_park_e,
            "r_park_m": self.r_park_m,
            "tof_max": self.tof_max,
            "tof_list": self._tof_list(),
        }

        for year in range(self.start_year, self.end_year + 1):
            jd0 = dt.date(year, 1, 1).toordinal() + 1721424.5
            jd1 = dt.date(year + 1, 1, 1).toordinal() + 1721424.5
            jd = jd0
            while jd < jd1:
                r = evaluate_day(self.eph, jd, opts)
                if r is not None:
                    self.results[jd] = r
                jd += 1.0
        print(f"[Sim] Готово: {len(self.results)} дат")

    def _pick_best(self):
        if not self.results:
            return
        successes = [r for r in self.results.values() if r.success]
        pool = successes if successes else list(self.results.values())
        best = min(pool, key=lambda r: r.dv)
        self.selected_jd = best.jd

    # ---------- отрисовка ----------
    def _refresh(self):
        self._draw_calendar()
        self._draw_sky()
        self._draw_info()
        self.fig.canvas.draw_idle()
        self.info_window.canvas.draw_idle()

    def _draw_calendar(self):
        n_years = self.end_year - self.start_year + 1
        grid = np.full((n_years, 366), -1.0)

        for jd, r in self.results.items():
            d = dt.date.fromordinal(int(jd - 1721424.5))
            if d.year < self.start_year or d.year > self.end_year:
                continue
            yi = d.year - self.start_year
            di = (d - dt.date(d.year, 1, 1)).days
            if r.success:
                grid[yi, di] = 0.2 + 0.8 * r.score
            else:
                excess = max(0.0, (r.dv - self.dv_budget) / 2.5)
                grid[yi, di] = -min(0.9, 0.3 + 0.6 * excess)

        self.cal_im.set_data(grid)
        self.cal_im.set_clim(-1.0, 1.0)

        for ln in list(self.ax_cal.lines):
            ln.remove()
        if self.selected_jd is not None:
            d = dt.date.fromordinal(int(self.selected_jd - 1721424.5))
            if self.start_year <= d.year <= self.end_year:
                yi = d.year - self.start_year
                di = (d - dt.date(d.year, 1, 1)).days
                self.ax_cal.plot(
                    [di, di + 1, di + 1, di, di],
                    [yi, yi, yi + 1, yi + 1, yi],
                    color="white", lw=1.4)

    def _draw_sky(self):
        jd = self.selected_jd
        if jd is None or jd not in self.results:
            return
        r = self.results[jd]

        e = r.earth_dep.pos
        m0 = r.mars_dep.pos
        m1 = r.mars_arr.pos

        self.earth_dot.set_data([e[0]], [e[1]])
        self.mars_dot.set_data([m0[0]], [m0[1]])
        self.mars_arr_dot.set_data([m1[0]], [m1[1]])

        pts = self._arc_points(r)
        self.traj_line.set_data(pts[:, 0], pts[:, 1])

        mid = pts[len(pts) // 2]
        self.sat_dot.set_data([mid[0]], [mid[1]])

    @staticmethod
    def _arc_points(r: DayResult, n: int = 160) -> np.ndarray:
        """Численное интегрирование траектории методом RK4 (2D-кеплер)."""
        mu = MU_SUN
        pos = r.r1.astype(float).copy()
        vel = r.v1.astype(float).copy()
        h = (r.tof * DAY_SEC) / n
        pts = [pos.copy()]

        def acc(p):
            rr = math.sqrt(p[0] ** 2 + p[1] ** 2)
            return -mu * p / rr ** 3

        for _ in range(n):
            k1v = acc(pos)
            k1r = vel
            k2v = acc(pos + 0.5 * h * k1r)
            k2r = vel + 0.5 * h * k1v
            k3v = acc(pos + 0.5 * h * k2r)
            k3r = vel + 0.5 * h * k2v
            k4v = acc(pos + h * k3r)
            k4r = vel + h * k3v
            pos = pos + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
            vel = vel + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
            pts.append(pos.copy())
        return np.array(pts)

    # ---------- таблица в отдельном окне ----------
    def _clear_info_artists(self):
        for a in self._info_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._info_artists.clear()

    def _draw_info(self):
        self._clear_info_artists()

        jd = self.selected_jd
        if jd is None or jd not in self.results:
            t = self.ax_info.text(
                0.5, 0.5, "Выберите дату в календаре главного окна.",
                transform=self.ax_info.transAxes,
                color="#8b98a8", fontsize=11, ha="center", va="center",
                family="monospace")
            self._info_artists.append(t)
            self.verdict_text.set_text("— ОЖИДАНИЕ —")
            self.verdict_text.set_color("#8b98a8")
            return

        r = self.results[jd]
        e_angle = math.atan2(r.earth_dep.y, r.earth_dep.x) * 180 / math.pi
        m_angle = math.atan2(r.mars_dep.y, r.mars_dep.x) * 180 / math.pi
        phase = ((m_angle - e_angle) % 360 + 360) % 360

        dep_date = dt.date.fromordinal(int(jd - 1721424.5))
        arr_date = dt.date.fromordinal(int(jd + r.tof - 1721424.5))

        rows = [
            ("СТАРТ", "Дата старта",              dep_date.isoformat(), ""),
            ("",      "Дата прибытия",            arr_date.isoformat(), ""),
            ("",      "Время полёта",             f"{r.tof:.0f}",       "сут"),
            ("",      "Фазовый угол на старте",   f"{phase:.1f}",       "°"),

            ("ГЕОМЕТРИЯ ПЕРЕЛЁТА",
             "v∞ отхода",                         f"{r.v_inf_dep:.3f}", "км/с"),
            ("",      "v∞ прилёта",               f"{r.v_inf_arr:.3f}", "км/с"),
            ("",      "C3 (характеристическая энергия)",
                                                   f"{r.c3:.2f}",        "км²/с²"),

            ("ИМПУЛЬСЫ",
             "Δv отхода от Земли",                f"{r.dv_dep:.3f}",    "км/с"),
            ("",      "Δv торможения у Марса",    f"{r.dv_arr:.3f}",    "км/с"),
            ("",      "Суммарная Δv",             f"{r.dv:.3f}",        "км/с"),
            ("",      "Бюджет Δv",                f"{self.dv_budget:.2f}", "км/с"),
        ]

        y_top = 0.885
        y_step = 0.058
        x_label = 0.06
        x_value = 0.62
        x_unit = 0.80

        y = y_top
        for section, label, value, unit in rows:
            if section:
                t = self.ax_info.text(
                    x_label, y, section,
                    transform=self.ax_info.transAxes,
                    color="#4fc3f7", fontsize=10, ha="left", va="center",
                    family="monospace", weight="bold")
                self._info_artists.append(t)
                y -= y_step * 0.8

                ln = self.ax_info.plot(
                    [x_label, 0.94], [y + 0.005, y + 0.005],
                    color="#22303d", lw=0.8,
                    transform=self.ax_info.transAxes)[0]
                self._info_artists.append(ln)
                y -= y_step * 0.4

            t_label = self.ax_info.text(
                x_label, y, label,
                transform=self.ax_info.transAxes,
                color="#8b98a8", fontsize=10, ha="left", va="center",
                family="monospace")
            t_value = self.ax_info.text(
                x_value, y, value,
                transform=self.ax_info.transAxes,
                color="#d8e2ec", fontsize=10, ha="right", va="center",
                family="monospace", weight="bold")
            t_unit = self.ax_info.text(
                x_unit, y, unit,
                transform=self.ax_info.transAxes,
                color="#8b98a8", fontsize=9, ha="left", va="center",
                family="monospace")
            self._info_artists.extend([t_label, t_value, t_unit])
            y -= y_step

        if r.success:
            self.verdict_text.set_text("✔ ЗАПУСК ВОЗМОЖЕН")
            self.verdict_text.set_color("#3cdc82")
        else:
            reasons = []
            if not r.reasons["dv"]:
                reasons.append(
                    f"Δv превышен на {r.dv - self.dv_budget:.3f} км/с")
            if not r.reasons["tof"]:
                reasons.append(
                    f"TOF {r.tof:.0f} сут > {self.tof_max:.0f} сут")
            self.verdict_text.set_text(
                "✖ ЗАПУСК НЕВОЗМОЖЕН\n" + "; ".join(reasons))
            self.verdict_text.set_color("#e05555")

    # ---------- клик по календарю ----------
    def _on_calendar_click(self, event):
        if event.inaxes is not self.ax_cal or event.xdata is None:
            return
        xi = int(event.xdata)
        yi = int(event.ydata)
        if not (0 <= yi < (self.end_year - self.start_year + 1)):
            return
        year = self.start_year + yi
        if not (1 <= xi <= 365):
            return
        d = dt.date(year, 1, 1) + dt.timedelta(days=xi - 1)
        jd = d.toordinal() + 1721424.5
        if jd in self.results:
            self.selected_jd = jd
            self._refresh()

    def run(self):
        plt.show()


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main():
    print("Симулятор межпланетных запусков Земля → Марс")
    print("=" * 50)
    print(f"Кэш: {CACHE_PATH}")

    client = HorizonsClient(CACHE_PATH)
    start = "2026-01-01"
    stop  = "2036-01-01"

    try:
        earth_rows = client.fetch("399", start, stop, step="2d")
        mars_rows  = client.fetch("499", start, stop, step="2d")
    except Exception as e:
        print(f"Не удалось получить эфемериды: {e}")
        print("Проверь интернет и доступ к ssd.jpl.nasa.gov")
        # Не падаем — используем грубый fallback на круговых орбитах
        earth_rows, mars_rows = _fallback_ephemeris(start, stop)

    eph = Ephemeris(earth_rows, mars_rows)
    print(f"Эфемериды: Земля {len(earth_rows)} точек, "
          f"Марс {len(mars_rows)} точек")

    sim = MarsSimulator(eph, start_year=2026, end_year=2035)
    sim.run()


def _fallback_ephemeris(start: str, stop: str):
    """Резервные эфемериды на круговых орбитах, если API недоступен."""
    d0 = dt.date.fromisoformat(start).toordinal() + 1721424.5
    d1 = dt.date.fromisoformat(stop).toordinal() + 1721424.5
    T_E = 365.256363 * DAY_SEC
    T_M = 686.980 * DAY_SEC
    R_E = 1.00000011 * AU
    R_M = 1.523679 * AU
    L0_E = 100.46435 * math.pi / 180
    L0_M = 355.45332 * math.pi / 180
    nE = TWO_PI / T_E
    nM = TWO_PI / T_M

    earth_rows, mars_rows = [], []
    jd = d0
    while jd <= d1:
        t = (jd - 2451545.0) * DAY_SEC
        thE = L0_E + nE * t
        thM = L0_M + nM * t
        earth_rows.append([jd,
                           R_E * math.cos(thE), R_E * math.sin(thE), 0.0,
                           -R_E * nE * math.sin(thE),
                            R_E * nE * math.cos(thE), 0.0])
        mars_rows.append([jd,
                          R_M * math.cos(thM), R_M * math.sin(thM), 0.0,
                          -R_M * nM * math.sin(thM),
                           R_M * nM * math.cos(thM), 0.0])
        jd += 2.0
    return earth_rows, mars_rows


if __name__ == "__main__":
    main()