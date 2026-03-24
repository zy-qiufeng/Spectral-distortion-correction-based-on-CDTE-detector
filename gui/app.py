"""
Tkinter GUI for the CdTe detector spectral distortion simulation.

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│  CdTe 探测器能谱失真仿真系统                                       │
├─────────────────────┬────────────────────────────────────────────┤
│  参数面板            │  能谱显示区域                               │
│  • 管电压 (kV)       │  ┌──────────────┐  ┌──────────────┐       │
│  • 管电流 (μA)       │  │  真实能谱     │  │  失真能谱     │       │
│  • 能谱数量          │  │              │  │              │       │
│  • 采集时间 (s)      │  └──────────────┘  └──────────────┘       │
│  [开始仿真]          │                                            │
│  进度条              │  信息面板                                   │
│                     │                                            │
└─────────────────────┴────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# Register a CJK-capable font for plot labels (searched in order, first hit wins)
_CJK_FONT_SEARCH = [
    # Linux (WenQuanYi)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
]
for _p in _CJK_FONT_SEARCH:
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
        _fname = _fm.FontProperties(fname=_p).get_name()
        matplotlib.rcParams["font.family"] = [_fname, "DejaVu Sans"]
        break

from simulation.event_stream import run_simulation, simulation_metadata
from simulation.charge_transport import BIAS_VOLTAGE

# ─── Colour palette ───────────────────────────────────────────────────────────
BG_DARK = "#1e1e2e"
BG_MID = "#2a2a3e"
BG_PANEL = "#313152"
FG_TEXT = "#cdd6f4"
ACCENT_BLUE = "#89b4fa"
ACCENT_GREEN = "#a6e3a1"
ACCENT_RED = "#f38ba8"
ACCENT_YELLOW = "#f9e2af"
ACCENT_MAUVE = "#cba6f7"


class SimulationApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CdTe 探测器能谱失真仿真系统")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1050, 680)

        # ── State variables ────────────────────────────────────────────────
        self._voltage_var = tk.DoubleVar(value=40.0)
        self._current_var = tk.DoubleVar(value=100.0)
        self._n_spectra_var = tk.IntVar(value=1)
        self._acq_time_var = tk.DoubleVar(value=1.0)
        self._bias_var = tk.DoubleVar(value=BIAS_VOLTAGE)
        self._progress_var = tk.DoubleVar(value=0.0)
        self._running = False

        # ── Layout ────────────────────────────────────────────────────────
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Main horizontal paned layout
        main_frame = tk.Frame(self, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: control panel
        left_panel = tk.Frame(main_frame, bg=BG_MID, width=260, relief=tk.FLAT,
                              bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left_panel.pack_propagate(False)
        self._build_control_panel(left_panel)

        # Right: plot + info
        right_panel = tk.Frame(main_frame, bg=BG_DARK)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot_area(right_panel)
        self._build_info_panel(right_panel)

    def _label(self, parent, text, bold=False, fg=FG_TEXT, size=10, **kw):
        font = ("Helvetica", size, "bold" if bold else "normal")
        return tk.Label(parent, text=text, bg=parent["bg"], fg=fg,
                        font=font, **kw)

    def _build_control_panel(self, parent: tk.Frame) -> None:
        # Title
        title = self._label(parent, "⚙ 仿真参数", bold=True,
                            fg=ACCENT_MAUVE, size=12)
        title.pack(pady=(12, 4))

        tk.Frame(parent, bg=ACCENT_MAUVE, height=2).pack(fill=tk.X,
                                                          padx=10, pady=4)

        # ── Voltage slider ─────────────────────────────────────────────
        self._add_slider_row(
            parent,
            label="管电压 (kV)",
            var=self._voltage_var,
            from_=20, to_=50, resolution=0.5,
            fmt="{:.1f} kV",
            accent=ACCENT_BLUE,
        )

        # ── Current slider ─────────────────────────────────────────────
        self._add_slider_row(
            parent,
            label="管电流 (μA)",
            var=self._current_var,
            from_=10, to_=200, resolution=1.0,
            fmt="{:.0f} μA",
            accent=ACCENT_GREEN,
        )

        # ── Bias voltage ───────────────────────────────────────────────
        self._add_slider_row(
            parent,
            label="探测器偏压 (V)",
            var=self._bias_var,
            from_=200, to_=1000, resolution=10,
            fmt="{:.0f} V",
            accent=ACCENT_YELLOW,
        )

        # ── Number of spectra ──────────────────────────────────────────
        sep = tk.Frame(parent, bg=BG_PANEL, height=1)
        sep.pack(fill=tk.X, padx=10, pady=6)

        row = tk.Frame(parent, bg=BG_MID)
        row.pack(fill=tk.X, padx=14, pady=4)
        self._label(row, "输出能谱数量", bold=True).pack(side=tk.LEFT)
        spin = tk.Spinbox(
            row, from_=1, to=20, textvariable=self._n_spectra_var,
            width=5, bg=BG_PANEL, fg=ACCENT_MAUVE,
            buttonbackground=BG_PANEL, relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
        )
        spin.pack(side=tk.RIGHT)

        # ── Acquisition time ───────────────────────────────────────────
        row2 = tk.Frame(parent, bg=BG_MID)
        row2.pack(fill=tk.X, padx=14, pady=4)
        self._label(row2, "采集时间 (s)").pack(side=tk.LEFT)
        acq_spin = tk.Spinbox(
            row2, from_=0.1, to=60.0, increment=0.1,
            textvariable=self._acq_time_var, width=6,
            bg=BG_PANEL, fg=ACCENT_YELLOW,
            buttonbackground=BG_PANEL, relief=tk.FLAT,
            font=("Helvetica", 10),
            format="%.1f",
        )
        acq_spin.pack(side=tk.RIGHT)

        # ── Run button ─────────────────────────────────────────────────
        tk.Frame(parent, bg=BG_PANEL, height=1).pack(fill=tk.X,
                                                      padx=10, pady=10)
        self._run_btn = tk.Button(
            parent, text="▶  开始仿真",
            command=self._on_run,
            bg=ACCENT_BLUE, fg=BG_DARK,
            activebackground=ACCENT_MAUVE, activeforeground=BG_DARK,
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, pady=8,
        )
        self._run_btn.pack(fill=tk.X, padx=14, pady=4)

        self._stop_btn = tk.Button(
            parent, text="■  停止",
            command=self._on_stop,
            bg=ACCENT_RED, fg=BG_DARK,
            activebackground="#fab387", activeforeground=BG_DARK,
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, pady=8,
            state=tk.DISABLED,
        )
        self._stop_btn.pack(fill=tk.X, padx=14, pady=2)

        # ── Progress bar ───────────────────────────────────────────────
        tk.Frame(parent, bg=BG_PANEL, height=1).pack(fill=tk.X,
                                                      padx=10, pady=8)
        self._label(parent, "仿真进度").pack(anchor=tk.W, padx=14)
        self._progress_bar = ttk.Progressbar(
            parent, variable=self._progress_var,
            maximum=100, mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill=tk.X, padx=14, pady=4)

        self._status_lbl = self._label(parent, "就绪", fg=ACCENT_GREEN)
        self._status_lbl.pack(anchor=tk.W, padx=14)

        # Style
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=BG_PANEL, background=ACCENT_BLUE,
            thickness=12,
        )

    def _add_slider_row(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.Variable,
        from_: float,
        to_: float,
        resolution: float,
        fmt: str,
        accent: str,
    ) -> None:
        """Add a labelled slider row to the control panel."""
        row_frame = tk.Frame(parent, bg=BG_MID)
        row_frame.pack(fill=tk.X, padx=14, pady=6)

        header = tk.Frame(row_frame, bg=BG_MID)
        header.pack(fill=tk.X)
        lbl = self._label(header, label, bold=True)
        lbl.pack(side=tk.LEFT)
        val_lbl = self._label(header, fmt.format(var.get()), fg=accent)
        val_lbl.pack(side=tk.RIGHT)

        def _update(v, vl=val_lbl, f=fmt, variable=var):
            vl.config(text=f.format(float(v)))

        slider = tk.Scale(
            row_frame, variable=var,
            from_=from_, to=to_, resolution=resolution,
            orient=tk.HORIZONTAL, showvalue=False,
            bg=BG_MID, fg=accent, troughcolor=BG_PANEL,
            activebackground=accent, highlightthickness=0,
            command=_update,
        )
        slider.pack(fill=tk.X)

    def _build_plot_area(self, parent: tk.Frame) -> None:
        plot_frame = tk.Frame(parent, bg=BG_DARK)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self._fig, (self._ax_true, self._ax_dist) = plt.subplots(
            1, 2, figsize=(11, 4.5),
            facecolor=BG_DARK,
        )
        self._style_axes()

        canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

        toolbar_frame = tk.Frame(plot_frame, bg=BG_DARK)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.config(bg=BG_MID)
        toolbar.update()

    def _style_axes(self) -> None:
        for ax, title, color in [
            (self._ax_true, "真实能谱 (True Spectrum)", ACCENT_GREEN),
            (self._ax_dist, "失真能谱 (Distorted Spectrum)", ACCENT_RED),
        ]:
            ax.set_facecolor(BG_PANEL)
            ax.set_title(title, color=color, fontsize=11, fontweight="bold")
            ax.set_xlabel("Energy (keV)", color=FG_TEXT, fontsize=9)
            ax.set_ylabel("Counts", color=FG_TEXT, fontsize=9)
            ax.tick_params(colors=FG_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BG_PANEL)
            ax.grid(True, color="#45475a", linewidth=0.5, alpha=0.6)

        self._fig.tight_layout(pad=2.0)

    def _build_info_panel(self, parent: tk.Frame) -> None:
        info_frame = tk.Frame(parent, bg=BG_MID, relief=tk.FLAT)
        info_frame.pack(fill=tk.X, padx=0, pady=(4, 0))

        self._label(info_frame, "📋 仿真参数信息", bold=True,
                    fg=ACCENT_YELLOW, size=10).pack(
            anchor=tk.W, padx=12, pady=(6, 2)
        )

        self._info_text = tk.Text(
            info_frame, height=4, bg=BG_PANEL, fg=FG_TEXT,
            font=("Courier", 9), relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self._info_text.pack(fill=tk.X, padx=12, pady=(0, 8))

    # ──────────────────────────────────────────────────────────────────────
    # Simulation control
    # ──────────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if self._running:
            return
        self._running = True
        self._run_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._progress_var.set(0)
        self._status_lbl.config(text="仿真运行中…", fg=ACCENT_YELLOW)

        # Read parameters
        voltage = self._voltage_var.get()
        current = self._current_var.get()
        n_spectra = self._n_spectra_var.get()
        acq_time = self._acq_time_var.get()
        bias = self._bias_var.get()

        # Show metadata immediately
        meta = simulation_metadata(voltage, current, acq_time, bias)
        self._update_info(meta)

        # Run in background thread to keep GUI responsive
        thread = threading.Thread(
            target=self._run_worker,
            args=(voltage, current, n_spectra, acq_time, bias),
            daemon=True,
        )
        thread.start()

    def _on_stop(self) -> None:
        self._running = False
        self._status_lbl.config(text="已停止", fg=ACCENT_RED)

    def _run_worker(
        self,
        voltage: float,
        current: float,
        n_spectra: int,
        acq_time: float,
        bias: float,
    ) -> None:
        """Runs simulation in background thread; posts results to GUI via after()."""
        try:
            def _progress(frac: float) -> None:
                if not self._running:
                    raise RuntimeError("stopped")
                self.after(0, self._progress_var.set, frac * 100)

            energy_bins, true_spec, dist_spec = run_simulation(
                voltage_kv=voltage,
                current_ua=current,
                n_spectra=n_spectra,
                acquisition_time_s=acq_time,
                bias_voltage=bias,
                progress_callback=_progress,
            )
            self.after(0, self._show_results, energy_bins, true_spec, dist_spec)
        except RuntimeError as exc:
            if "stopped" not in str(exc):
                self.after(0, messagebox.showerror, "仿真错误", str(exc))
        except Exception as exc:
            self.after(0, messagebox.showerror, "仿真错误", str(exc))
        finally:
            self.after(0, self._finish_run)

    def _finish_run(self) -> None:
        self._running = False
        self._run_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        if self._status_lbl.cget("text") == "仿真运行中…":
            self._status_lbl.config(text="仿真完成 ✓", fg=ACCENT_GREEN)

    def _show_results(
        self,
        energy_bins: np.ndarray,
        true_spec: np.ndarray,
        dist_spec: np.ndarray,
    ) -> None:
        """Update the matplotlib plots with new simulation results."""
        for ax in (self._ax_true, self._ax_dist):
            ax.cla()
        self._style_axes()

        # True spectrum
        self._ax_true.fill_between(
            energy_bins, true_spec, alpha=0.35, color=ACCENT_GREEN,
        )
        self._ax_true.plot(energy_bins, true_spec, color=ACCENT_GREEN,
                           linewidth=1.2, label="True")
        self._ax_true.set_xlim(energy_bins[0], energy_bins[-1])
        self._ax_true.set_ylim(bottom=0)
        self._ax_true.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT,
                             fontsize=8, framealpha=0.8)

        # Distorted spectrum
        self._ax_dist.fill_between(
            energy_bins, dist_spec, alpha=0.35, color=ACCENT_RED,
        )
        self._ax_dist.plot(energy_bins, dist_spec, color=ACCENT_RED,
                           linewidth=1.2, label="Distorted")
        # Overlay true spectrum as ghost for comparison
        self._ax_dist.plot(energy_bins, true_spec, color=ACCENT_GREEN,
                           linewidth=0.8, linestyle="--", alpha=0.6,
                           label="True (ref.)")
        self._ax_dist.set_xlim(energy_bins[0], energy_bins[-1])
        self._ax_dist.set_ylim(bottom=0)
        self._ax_dist.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT,
                             fontsize=8, framealpha=0.8)

        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()

    def _update_info(self, meta: dict) -> None:
        """Populate the info text box with simulation metadata."""
        lines = [
            f"管电压: {meta['tube_voltage_kv']:.1f} kV   "
            f"管电流: {meta['tube_current_ua']:.0f} μA",
            f"偏压: {meta['bias_voltage_V']:.0f} V   "
            f"有效偏压 (极化后): {meta['effective_bias_V']:.1f} V   "
            f"极化率: {meta['polarisation_fraction']*100:.1f}%",
            f"光子通量: {meta['photon_rate_cps']:.1f} cps   "
            f"预期计数: {meta['expected_counts']:.0f}   "
            f"采集时间: {meta['acquisition_time_s']:.1f} s",
        ]
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, "\n".join(lines))
        self._info_text.config(state=tk.DISABLED)
