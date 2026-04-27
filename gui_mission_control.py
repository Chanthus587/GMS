"""
gui_mission_control.py
══════════════════════════════════════════════════════════════════════
GMS Mission Control Dashboard
Gradient-Momentum Microclimate Instability Detection System

Run:
    python gui_mission_control.py

Controls:
    SPACE       — play / pause simulation
    R           — reset to t=0
    LEFT/RIGHT  — step one frame back / forward
    S           — save current frame as PNG
    Q / Escape  — quit
══════════════════════════════════════════════════════════════════════
"""

import sys
import os
import math
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d

# ══════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (dark mission-control theme)
# ══════════════════════════════════════════════════════════════════════
BG_DARK   = "#0D1117"   # window background
BG_PANEL  = "#161B22"   # panel background
BG_CARD   = "#1C2128"   # card / widget background
BG_HOVER  = "#21262D"
FG_PRIMARY   = "#E6EDF3"
FG_SECONDARY = "#8B949E"
FG_TERTIARY  = "#484F58"
ACCENT_BLUE  = "#388BFD"
ACCENT_TEAL  = "#3FB950"
ACCENT_AMBER = "#D29922"
ACCENT_RED   = "#F85149"
ACCENT_PURPLE= "#BC8CFF"
BORDER       = "#30363D"

# GMS colourmap: blue → amber → red
GMS_CMAP = LinearSegmentedColormap.from_list(
    "gms_dark",
    ["#0D4F8C","#1565C0","#1976D2","#388BFD",
     "#D29922","#E65100","#F85149","#C62828"],
    N=256
)

NODE_STABLE  = "#388BFD"
NODE_MOD     = "#D29922"
NODE_HIGH    = "#F85149"

EVENT_COLS = [ACCENT_BLUE, ACCENT_TEAL, ACCENT_AMBER]

# ══════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE  (self-contained, no external config needed)
# ══════════════════════════════════════════════════════════════════════

class SimEngine:
    """Runs the full GMS pipeline and exposes per-timestep results."""

    N_NODES  = 12
    T_STEPS  = 120
    GRID     = 10.0
    RADIUS   = 3.8
    SEED     = 2024

    W1, W2, W3, W4 = 0.35, 0.25, 0.20, 0.20
    THETA  = 1.2
    WINDOW = 8
    ALPHA  = 0.30
    BETA   = 0.60
    ABS_THRESH = 26.5

    EVENTS = [
        dict(nodes=[0,1,2], t_start=25, t_end=55, dT=7.0, label="Event A"),
        dict(nodes=[8,9],   t_start=38, t_end=70, dT=4.5, label="Event B"),
        dict(nodes=[3,4,5], t_start=55, t_end=90, dT=5.5, label="Event C"),
    ]

    def __init__(self):
        np.random.seed(self.SEED)
        self.pos   = self._place_nodes()
        self.dist  = cdist(self.pos, self.pos)
        self.adj   = {i: [j for j in range(self.N_NODES)
                          if j != i and self.dist[i,j] <= self.RADIUS]
                      for i in range(self.N_NODES)}
        self.Temp, self.Humid = self._simulate()
        self._run_gms()

    def _place_nodes(self):
        pos = np.random.uniform(0.8, self.GRID - 0.8, (self.N_NODES, 2))
        pins = [(1.5,1.5),(2.2,1.8),(1.8,2.5),
                (5.0,5.0),(5.8,4.8),(5.3,5.9),
                (8.0,7.5),(8.6,8.0)]
        for i,(x,y) in enumerate(pins[:self.N_NODES]):
            pos[i] = [x, y]
        return pos

    def _simulate(self):
        t  = np.linspace(0, 2*np.pi, self.T_STEPS)
        T  = np.zeros((self.N_NODES, self.T_STEPS))
        H  = np.zeros((self.N_NODES, self.T_STEPS))
        for i in range(self.N_NODES):
            elev   = (self.pos[i,0]+self.pos[i,1])/(2*self.GRID)
            T[i]   = 22 + 6*np.sin(t-0.3) + elev*2.5 + np.random.normal(0,0.25,self.T_STEPS)
            H[i]   = 65 -  8*np.sin(t)    - elev*3.0 + np.random.normal(0,0.50,self.T_STEPS)
        for ev in self.EVENTS:
            dur = ev['t_end'] - ev['t_start']
            for i in ev['nodes']:
                ramp = np.zeros(self.T_STEPS)
                ramp[ev['t_start']:ev['t_end']] = np.linspace(0, ev['dT'], dur)
                ramp[ev['t_end']:]              = ev['dT']*np.exp(-np.arange(self.T_STEPS-ev['t_end'])/12.)
                T[i] += ramp
                H[i] -= ramp*0.8
        return T, H

    def _run_gms(self):
        N, T = self.N_NODES, self.T_STEPS
        # Gradient
        G = np.zeros((N,T))
        for i in range(N):
            nb = self.adj[i]
            if nb:
                G[i] = np.array([self.Temp[i]-self.Temp[j] for j in nb]).mean(0)
        self.grad = G
        # Momentum
        M = np.zeros((N,T))
        M[:,1:] = G[:,1:] - G[:,:-1]
        self.mom = M
        # Duration
        D = np.zeros((N,T))
        for i in range(N):
            for t in range(T):
                ws = max(0, t-self.WINDOW+1)
                D[i,t] = np.mean(np.abs(G[i,ws:t+1]) > self.THETA)
        self.dur = D
        # NIS
        NIS = np.zeros((N,T))
        for i in range(N):
            nb = self.adj[i]
            if nb:
                NIS[i] = np.array([self.Temp[i]-self.Temp[j] for j in nb]).mean(0)
        mn,mx = NIS.min(), NIS.max()
        self.nis = (NIS-mn)/(mx-mn+1e-12)
        # GMS
        def norm(x):
            mn,mx = x.min(), x.max(); return (x-mn)/(mx-mn+1e-12)
        raw = (self.W1*norm(np.abs(G)) + self.W2*norm(np.abs(M))
             + self.W3*self.nis        + self.W4*D)
        self.gms = np.clip(norm(raw), 0, 1)
        # Labels
        lbl = np.zeros((N,T), dtype=int)
        lbl[self.gms >= self.ALPHA] = 1
        lbl[self.gms >= self.BETA]  = 2
        self.label = lbl
        # Baseline alarm
        self.baseline = (self.Temp > self.ABS_THRESH).astype(int)
        # Onset times
        self.onset = np.full(N, np.inf)
        for i in range(N):
            above = np.where(self.gms[i] > self.ALPHA)[0]
            if len(above): self.onset[i] = above[0]

    def rerun(self, w1, w2, w3, w4, theta, window, alpha, beta):
        """Recompute GMS with new weights — called from GUI sliders."""
        self.W1,self.W2,self.W3,self.W4 = w1,w2,w3,w4
        self.THETA = theta; self.WINDOW = int(window)
        self.ALPHA = alpha; self.BETA   = beta
        self._run_gms()


# ══════════════════════════════════════════════════════════════════════
#  MISSION CONTROL GUI
# ══════════════════════════════════════════════════════════════════════

class MissionControlGUI:

    ANIM_INTERVAL_MS = 120   # ms between frames during playback

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.engine = SimEngine()
        self.t      = 0
        self.playing = False
        self.selected_node = 0
        self._anim_job = None

        self._build_window()
        self._build_layout()
        self._draw_all(self.t)
        self._bind_keys()

    # ──────────────────────────────────────────────────────────────────
    #  WINDOW SETUP
    # ──────────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root.title("GMS Mission Control  —  Microclimate Instability Detection")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1380, 820)
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1400x860")

    def _build_layout(self):
        """
        ┌───────────── TOP BAR ──────────────────────────────────┐
        │ LEFT PANEL │     CENTRE CANVAS     │  RIGHT PANEL       │
        │  controls  │  sensor map + charts  │  alerts + metrics  │
        └────────────────────────────────────────────────────────┘
        """
        # ── top bar ──────────────────────────────────────────────
        self._build_topbar()

        # ── main body ────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=8, pady=(0,8))

        body.columnconfigure(0, minsize=230, weight=0)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, minsize=250, weight=0)
        body.rowconfigure(0, weight=1)

        left  = tk.Frame(body, bg=BG_PANEL, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))

        centre = tk.Frame(body, bg=BG_DARK)
        centre.grid(row=0, column=1, sticky="nsew", padx=(0,6))

        right = tk.Frame(body, bg=BG_PANEL, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=2, sticky="nsew")

        self._build_left_panel(left)
        self._build_centre(centre)
        self._build_right_panel(right)

    # ── TOP BAR ──────────────────────────────────────────────────────

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL,
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x", padx=8, pady=(8,6))

        # Title
        tk.Label(bar, text="GMS MISSION CONTROL",
                 bg=BG_PANEL, fg=FG_PRIMARY,
                 font=("Courier New", 14, "bold")).pack(side="left", padx=16, pady=8)
        tk.Label(bar, text="Gradient-Momentum Microclimate Instability Detection",
                 bg=BG_PANEL, fg=FG_SECONDARY,
                 font=("Courier New", 9)).pack(side="left", padx=4)

        # Playback controls (right side)
        ctrl = tk.Frame(bar, bg=BG_PANEL)
        ctrl.pack(side="right", padx=14, pady=6)

        btn_kw = dict(bg=BG_CARD, fg=FG_PRIMARY,
                      activebackground=BG_HOVER, activeforeground=FG_PRIMARY,
                      relief="flat", bd=0, padx=10, pady=4,
                      font=("Courier New", 10, "bold"), cursor="hand2")

        tk.Button(ctrl, text="◀◀", command=self._step_back,   **btn_kw).pack(side="left", padx=2)
        self.btn_play = tk.Button(ctrl, text="▶  PLAY",
                                  command=self._toggle_play, **btn_kw)
        self.btn_play.pack(side="left", padx=2)
        tk.Button(ctrl, text="▶▶", command=self._step_fwd,    **btn_kw).pack(side="left", padx=2)
        tk.Button(ctrl, text="↺  RESET", command=self._reset, **btn_kw).pack(side="left", padx=8)

        # Speed
        tk.Label(ctrl, text="Speed", bg=BG_PANEL, fg=FG_SECONDARY,
                 font=("Courier New", 9)).pack(side="left", padx=(8,2))
        self.speed_var = tk.IntVar(value=120)
        speed_sl = ttk.Scale(ctrl, from_=30, to=400, variable=self.speed_var,
                             orient="horizontal", length=90,
                             command=lambda v: None)
        speed_sl.pack(side="left", padx=2)

        # Time display
        self.lbl_time = tk.Label(bar, text="t = 000",
                                  bg=BG_PANEL, fg=ACCENT_BLUE,
                                  font=("Courier New", 12, "bold"))
        self.lbl_time.pack(side="right", padx=24)

        # Save button
        tk.Button(bar, text="⤓ SAVE", command=self._save_figure,
                  **btn_kw).pack(side="right", padx=6)

    # ── LEFT PANEL (controls + weights) ──────────────────────────────

    def _build_left_panel(self, parent):
        parent.columnconfigure(0, weight=1)

        self._section(parent, "SIMULATION CONTROLS", row=0)

        # Timeline slider
        self._label(parent, "Timeline", row=1)
        self.t_var = tk.IntVar(value=0)
        self.t_slider = ttk.Scale(parent, from_=0,
                                   to=self.engine.T_STEPS-1,
                                   variable=self.t_var, orient="horizontal",
                                   command=self._on_slider)
        self.t_slider.grid(row=2, column=0, sticky="ew", padx=12, pady=(0,8))

        # Event buttons
        ev_frame = tk.Frame(parent, bg=BG_PANEL)
        ev_frame.grid(row=3, column=0, sticky="ew", padx=12, pady=(0,10))
        for idx, ev in enumerate(self.engine.EVENTS):
            col = [ACCENT_BLUE, ACCENT_TEAL, ACCENT_AMBER][idx]
            tk.Button(ev_frame, text=f"Jump {ev['label']}",
                      bg=BG_CARD, fg=col,
                      activebackground=BG_HOVER, activeforeground=col,
                      relief="flat", bd=0, padx=6, pady=3,
                      font=("Courier New", 8, "bold"), cursor="hand2",
                      command=lambda t=ev['t_start']: self._jump_to(t)
                      ).pack(side="left", padx=(0,4))

        self._section(parent, "GMS WEIGHTS", row=4)

        # Weight sliders
        self.w_vars = {}
        labels = [("w₁  Gradient",  "W1", 0.35, ACCENT_BLUE),
                  ("w₂  Momentum",  "W2", 0.25, ACCENT_TEAL),
                  ("w₃  NIS",       "W3", 0.20, ACCENT_PURPLE),
                  ("w₄  Duration",  "W4", 0.20, ACCENT_AMBER)]
        for r_off, (lbl, key, default, col) in enumerate(labels):
            row = 5 + r_off * 2
            tk.Label(parent, text=lbl, bg=BG_PANEL, fg=col,
                     font=("Courier New", 9, "bold")
                     ).grid(row=row, column=0, sticky="w", padx=12, pady=(4,0))
            var = tk.DoubleVar(value=default)
            self.w_vars[key] = var
            frm = tk.Frame(parent, bg=BG_PANEL)
            frm.grid(row=row+1, column=0, sticky="ew", padx=12, pady=(0,2))
            frm.columnconfigure(0, weight=1)
            sl = ttk.Scale(frm, from_=0.05, to=0.70, variable=var,
                           orient="horizontal",
                           command=lambda v, k=key: self._on_weight(k))
            sl.grid(row=0, column=0, sticky="ew")
            lbl_val = tk.Label(frm, textvariable=var, bg=BG_PANEL,
                               fg=FG_SECONDARY, font=("Courier New", 8),
                               width=4)
            lbl_val.grid(row=0, column=1, padx=(4,0))

        self._section(parent, "THRESHOLDS", row=13)

        thresh_items = [("θ  Theta",  "THETA",  1.2, 0.2, 3.0,  ACCENT_TEAL),
                        ("α  Stable", "ALPHA",  0.30, 0.1, 0.5, ACCENT_AMBER),
                        ("β  High",   "BETA",   0.60, 0.4, 0.9, ACCENT_RED)]
        self.th_vars = {}
        for r_off, (lbl, key, default, lo, hi, col) in enumerate(thresh_items):
            row = 14 + r_off * 2
            tk.Label(parent, text=lbl, bg=BG_PANEL, fg=col,
                     font=("Courier New", 9, "bold")
                     ).grid(row=row, column=0, sticky="w", padx=12, pady=(4,0))
            var = tk.DoubleVar(value=default)
            self.th_vars[key] = var
            frm = tk.Frame(parent, bg=BG_PANEL)
            frm.grid(row=row+1, column=0, sticky="ew", padx=12, pady=(0,2))
            frm.columnconfigure(0, weight=1)
            sl = ttk.Scale(frm, from_=lo, to=hi, variable=var,
                           orient="horizontal",
                           command=lambda v, k=key: self._on_threshold(k))
            sl.grid(row=0, column=0, sticky="ew")
            tk.Label(frm, textvariable=var, bg=BG_PANEL,
                     fg=FG_SECONDARY, font=("Courier New", 8),
                     width=4).grid(row=0, column=1, padx=(4,0))

        self._section(parent, "NODE SELECTOR", row=20)
        node_frm = tk.Frame(parent, bg=BG_PANEL)
        node_frm.grid(row=21, column=0, sticky="ew", padx=12, pady=(4,8))
        for i in range(self.engine.N_NODES):
            r, c = divmod(i, 4)
            btn = tk.Button(node_frm, text=f"N{i}", width=3,
                            bg=BG_CARD, fg=FG_PRIMARY, relief="flat", bd=0,
                            font=("Courier New", 8), cursor="hand2",
                            activebackground=BG_HOVER,
                            command=lambda n=i: self._select_node(n))
            btn.grid(row=r, column=c, padx=2, pady=2)

        parent.rowconfigure(22, weight=1)

    # ── CENTRE (sensor map + time-series charts) ──────────────────────

    def _build_centre(self, parent):
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)
        parent.columnconfigure(0, weight=1)

        # Top: sensor map figure
        self.fig_map = plt.Figure(figsize=(7, 4.5), facecolor=BG_DARK)
        self.ax_map  = self.fig_map.add_subplot(111, facecolor=BG_PANEL)
        self._style_ax(self.ax_map)
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=parent)
        self.canvas_map.get_tk_widget().grid(row=0, column=0,
                                              sticky="nsew", pady=(0,4))
        self.canvas_map.mpl_connect("button_press_event", self._on_map_click)

        # Bottom: dual time-series (GMS + Temperature)
        self.fig_ts = plt.Figure(figsize=(7, 2.8), facecolor=BG_DARK)
        gs = gridspec.GridSpec(2, 1, figure=self.fig_ts,
                               hspace=0.08, left=0.07, right=0.97,
                               top=0.93, bottom=0.12)
        self.ax_gms_ts = self.fig_ts.add_subplot(gs[0], facecolor=BG_PANEL)
        self.ax_tmp_ts = self.fig_ts.add_subplot(gs[1], facecolor=BG_PANEL)
        for ax in [self.ax_gms_ts, self.ax_tmp_ts]:
            self._style_ax(ax)
        self.canvas_ts = FigureCanvasTkAgg(self.fig_ts, master=parent)
        self.canvas_ts.get_tk_widget().grid(row=1, column=0,
                                             sticky="nsew")

    # ── RIGHT PANEL (alerts + live metrics + heatmap strip) ──────────

    def _build_right_panel(self, parent):
        parent.columnconfigure(0, weight=1)

        self._section(parent, "LIVE METRICS", row=0)

        # Metric cards grid
        metrics_frame = tk.Frame(parent, bg=BG_PANEL)
        metrics_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(4,10))
        metrics_frame.columnconfigure((0,1), weight=1)

        self.metric_cards = {}
        card_defs = [
            ("GMS Score",  "gms",   ACCENT_BLUE,   "0.000"),
            ("Gradient",   "grad",  ACCENT_TEAL,   "0.00°"),
            ("Momentum",   "mom",   ACCENT_PURPLE,  "0.00°"),
            ("NIS",        "nis",   ACCENT_AMBER,  "0.000"),
            ("Duration",   "dur",   ACCENT_RED,    "0.000"),
            ("Temp (°C)",  "temp",  FG_SECONDARY,  "00.0°"),
        ]
        for idx, (label, key, col, default) in enumerate(card_defs):
            r, c = divmod(idx, 2)
            card = tk.Frame(metrics_frame, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=r, column=c, sticky="ew", padx=3, pady=3, ipady=6)
            tk.Label(card, text=label, bg=BG_CARD, fg=FG_SECONDARY,
                     font=("Courier New", 8)).pack(pady=(4,0))
            val_lbl = tk.Label(card, text=default, bg=BG_CARD, fg=col,
                               font=("Courier New", 14, "bold"))
            val_lbl.pack()
            self.metric_cards[key] = val_lbl

        # Status badge
        self._section(parent, "NODE STATUS", row=2)
        self.status_badge = tk.Label(parent, text="● STABLE",
                                      bg=BG_PANEL, fg=ACCENT_TEAL,
                                      font=("Courier New", 14, "bold"))
        self.status_badge.grid(row=3, column=0, pady=(4,8))
        self.lbl_node = tk.Label(parent, text="Inspecting: N0",
                                  bg=BG_PANEL, fg=FG_SECONDARY,
                                  font=("Courier New", 9))
        self.lbl_node.grid(row=4, column=0, pady=(0,8))

        # Alert log
        self._section(parent, "ALERT LOG", row=5)
        log_frame = tk.Frame(parent, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=(4,10))
        self.alert_text = tk.Text(log_frame, bg=BG_CARD, fg=FG_SECONDARY,
                                   font=("Courier New", 8),
                                   height=7, bd=0, wrap="word",
                                   state="disabled",
                                   insertbackground=FG_PRIMARY)
        self.alert_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.alert_text.tag_config("high",  foreground=ACCENT_RED)
        self.alert_text.tag_config("mod",   foreground=ACCENT_AMBER)
        self.alert_text.tag_config("info",  foreground=ACCENT_BLUE)
        self.alert_text.tag_config("ok",    foreground=ACCENT_TEAL)

        # Mini heatmap strip
        self._section(parent, "INSTABILITY HEATMAP", row=7)
        self.fig_heat = plt.Figure(figsize=(2.8, 2.2), facecolor=BG_DARK)
        self.ax_heat  = self.fig_heat.add_subplot(111, facecolor=BG_PANEL)
        self._style_ax(self.ax_heat)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=parent)
        self.canvas_heat.get_tk_widget().grid(row=8, column=0,
                                               sticky="ew", padx=8, pady=(4,8))

        # Performance table
        self._section(parent, "PERFORMANCE vs BASELINE", row=9)
        perf_frame = tk.Frame(parent, bg=BG_CARD,
                              highlightbackground=BORDER, highlightthickness=1)
        perf_frame.grid(row=10, column=0, sticky="ew", padx=10, pady=(4,10))
        self.perf_labels = {}
        for r_idx, (metric, col) in enumerate([
                ("Accuracy",  ACCENT_TEAL),
                ("Precision", ACCENT_BLUE),
                ("Recall",    ACCENT_PURPLE),
                ("FAR",       ACCENT_AMBER)]):
            tk.Label(perf_frame, text=metric, bg=BG_CARD, fg=FG_SECONDARY,
                     font=("Courier New", 8), width=10, anchor="w"
                     ).grid(row=r_idx, column=0, padx=6, pady=2, sticky="w")
            lbl = tk.Label(perf_frame, text="—", bg=BG_CARD, fg=col,
                           font=("Courier New", 9, "bold"), width=7, anchor="e")
            lbl.grid(row=r_idx, column=1, padx=6, pady=2, sticky="e")
            self.perf_labels[metric] = lbl

        parent.rowconfigure(11, weight=1)

    # ── HELPERS ──────────────────────────────────────────────────────

    def _section(self, parent, text, row):
        frm = tk.Frame(parent, bg=BORDER, height=1)
        frm.grid(row=row, column=0, sticky="ew", padx=8, pady=(8,0))
        tk.Label(parent, text=text, bg=BG_PANEL, fg=FG_TERTIARY,
                 font=("Courier New", 8, "bold")
                 ).grid(row=row, column=0, sticky="w", padx=12)

    def _label(self, parent, text, row):
        tk.Label(parent, text=text, bg=BG_PANEL, fg=FG_SECONDARY,
                 font=("Courier New", 9)).grid(
                     row=row, column=0, sticky="w", padx=12, pady=(6,2))

    def _style_ax(self, ax):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=FG_TERTIARY, labelsize=7)
        ax.xaxis.label.set_color(FG_SECONDARY)
        ax.yaxis.label.set_color(FG_SECONDARY)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

    def _bind_keys(self):
        self.root.bind("<space>",  lambda e: self._toggle_play())
        self.root.bind("<r>",      lambda e: self._reset())
        self.root.bind("<R>",      lambda e: self._reset())
        self.root.bind("<Left>",   lambda e: self._step_back())
        self.root.bind("<Right>",  lambda e: self._step_fwd())
        self.root.bind("<s>",      lambda e: self._save_figure())
        self.root.bind("<q>",      lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # ══════════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════════

    def _draw_all(self, t):
        self._draw_map(t)
        self._draw_timeseries(t)
        self._draw_heatmap(t)
        self._update_metrics(t)
        self._update_alerts(t)
        self._update_performance()
        self.lbl_time.config(text=f"t = {t:03d}")
        self.t_var.set(t)

    # ── SENSOR MAP ───────────────────────────────────────────────────

    def _draw_map(self, t):
        ax = self.ax_map
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        self._style_ax(ax)

        eng  = self.engine
        gms  = eng.gms[:, t]
        lbl  = eng.label[:, t]
        grad = eng.grad[:, t]

        # Grid lines (subtle)
        for v in np.arange(0, eng.GRID+1, 2):
            ax.axhline(v, color=BORDER, lw=0.3, alpha=0.4)
            ax.axvline(v, color=BORDER, lw=0.3, alpha=0.4)

        # Adjacency edges — colour by gradient magnitude
        for i in range(eng.N_NODES):
            for j in eng.adj[i]:
                if j > i:
                    gval = (abs(grad[i]) + abs(grad[j])) / 2
                    intensity = min(1.0, gval / 6.0)
                    r = int(13  + (248-13)  * intensity)
                    g = int(17  + (81-17)   * intensity)
                    b = int(23  + (73-23)   * intensity)
                    edge_col = f"#{r:02x}{g:02x}{b:02x}"
                    ax.plot([eng.pos[i,0], eng.pos[j,0]],
                            [eng.pos[i,1], eng.pos[j,1]],
                            color=edge_col, lw=1.0+intensity*2, alpha=0.6,
                            zorder=1)

        # Propagation arrows — draw if instability spreading
        for i in range(eng.N_NODES):
            for j in eng.adj[i]:
                if j > i:
                    if (np.isfinite(eng.onset[i]) and
                            np.isfinite(eng.onset[j]) and
                            0 < eng.onset[j] - eng.onset[i] <= 15 and
                            t >= eng.onset[i]):
                        xi,yi = eng.pos[i]; xj,yj = eng.pos[j]
                        ax.annotate("",
                            xy=(xi+(xj-xi)*0.78, yi+(yj-yi)*0.78),
                            xytext=(xi+(xj-xi)*0.22, yi+(yj-yi)*0.22),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color=ACCENT_AMBER,
                                lw=1.4, mutation_scale=10,
                                alpha=0.7
                            ), zorder=3)

        # Instability halo (pulse effect for unstable nodes)
        for i in range(eng.N_NODES):
            if lbl[i] >= 1:
                halo_r = 0.25 + gms[i] * 0.55
                col = NODE_MOD if lbl[i] == 1 else NODE_HIGH
                circle = plt.Circle(eng.pos[i], halo_r,
                                    color=col, alpha=0.15, zorder=2)
                ax.add_patch(circle)
                if lbl[i] == 2:
                    outer = plt.Circle(eng.pos[i], halo_r*1.5,
                                       color=col, alpha=0.06, zorder=1)
                    ax.add_patch(outer)

        # Nodes
        node_sizes  = 180 + gms * 320
        node_colors = []
        for i in range(eng.N_NODES):
            if lbl[i] == 2:   node_colors.append(NODE_HIGH)
            elif lbl[i] == 1: node_colors.append(NODE_MOD)
            else:             node_colors.append(NODE_STABLE)

        sc = ax.scatter(eng.pos[:,0], eng.pos[:,1],
                        s=node_sizes, c=node_colors,
                        zorder=4, edgecolors="white", linewidths=0.8)

        # Selected node highlight
        sn = self.selected_node
        ax.scatter(eng.pos[sn,0], eng.pos[sn,1],
                   s=node_sizes[sn]+120, marker="o",
                   facecolors="none", edgecolors="white",
                   linewidths=2.0, zorder=5)

        # Node labels
        for i in range(eng.N_NODES):
            ax.text(eng.pos[i,0]+0.15, eng.pos[i,1]+0.18,
                    f"N{i}\n{gms[i]:.2f}",
                    fontsize=6.5, color=FG_PRIMARY,
                    fontweight="bold", zorder=6,
                    fontfamily="monospace")

        # Event region shading
        for ev_idx, ev in enumerate(eng.EVENTS):
            if ev['t_start'] <= t < ev['t_end']:
                col = EVENT_COLS[ev_idx]
                for ni in ev['nodes']:
                    ax.add_patch(plt.Circle(
                        eng.pos[ni], 0.8, color=col,
                        alpha=0.08, zorder=0, linestyle='--',
                        fill=True))

        # Legend
        legend_items = [
            mpatches.Patch(color=NODE_STABLE, label="Stable"),
            mpatches.Patch(color=NODE_MOD,    label="Mod. Unstable"),
            mpatches.Patch(color=NODE_HIGH,   label="High Unstable"),
        ]
        ax.legend(handles=legend_items, loc="upper right",
                  facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=FG_SECONDARY, fontsize=7.5,
                  framealpha=0.9)

        ax.set_xlim(-0.4, eng.GRID+0.4)
        ax.set_ylim(-0.4, eng.GRID+0.4)
        ax.set_xlabel("X position (km)", fontsize=8)
        ax.set_ylabel("Y position (km)", fontsize=8)
        ax.set_title(f"Sensor Network  ·  t = {t:03d}  ·  "
                     f"Node N{sn} selected",
                     color=FG_PRIMARY, fontsize=9,
                     fontweight="bold", fontfamily="monospace", pad=8)

        self.canvas_map.draw_idle()

    # ── TIME-SERIES CHARTS ────────────────────────────────────────────

    def _draw_timeseries(self, t):
        eng = self.engine
        sn  = self.selected_node
        ts  = np.arange(eng.T_STEPS)

        # ── GMS time-series ───────────────────────────────────────
        ax1 = self.ax_gms_ts
        ax1.clear(); ax1.set_facecolor(BG_PANEL); self._style_ax(ax1)

        ax1.fill_between(ts, 0, eng.gms[sn],
                         alpha=0.18, color=ACCENT_BLUE)
        ax1.plot(ts, eng.gms[sn],
                 color=ACCENT_BLUE, lw=1.5, zorder=3,
                 label=f"GMS N{sn}")
        # Other nodes faint
        for i in range(eng.N_NODES):
            if i != sn:
                ax1.plot(ts, eng.gms[i],
                         color=FG_TERTIARY, lw=0.4, alpha=0.35)

        ax1.axhline(eng.ALPHA, color=ACCENT_AMBER, lw=0.9,
                    ls="--", alpha=0.8)
        ax1.axhline(eng.BETA,  color=ACCENT_RED,   lw=0.9,
                    ls="--", alpha=0.8)
        ax1.axvline(t, color="white", lw=1.2, alpha=0.5, zorder=4)
        ax1.scatter([t], [eng.gms[sn,t]],
                    s=60, c=ACCENT_BLUE, zorder=5, edgecolors="white",
                    linewidths=0.8)

        # Event bands
        for ev_idx, ev in enumerate(eng.EVENTS):
            ax1.axvspan(ev['t_start'], ev['t_end'],
                        alpha=0.06, color=EVENT_COLS[ev_idx])

        ax1.set_ylabel("GMS", fontsize=7, color=FG_SECONDARY)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, eng.T_STEPS-1)
        ax1.tick_params(labelbottom=False)
        ax1.set_title(f"GMS time-series  ·  Node N{sn}  (others faded)",
                      color=FG_SECONDARY, fontsize=7.5,
                      fontfamily="monospace", loc="left", pad=3)

        # ── Temperature time-series ───────────────────────────────
        ax2 = self.ax_tmp_ts
        ax2.clear(); ax2.set_facecolor(BG_PANEL); self._style_ax(ax2)

        ax2.fill_between(ts, eng.Temp[sn],
                         eng.ABS_THRESH,
                         where=eng.Temp[sn] > eng.ABS_THRESH,
                         alpha=0.20, color=ACCENT_RED)
        ax2.plot(ts, eng.Temp[sn],
                 color=ACCENT_TEAL, lw=1.4, label=f"Temp N{sn}")
        ax2.axhline(eng.ABS_THRESH, color=ACCENT_RED, lw=0.9,
                    ls="--", alpha=0.8, label="Baseline thresh")
        ax2.axvline(t, color="white", lw=1.2, alpha=0.5, zorder=4)
        ax2.scatter([t], [eng.Temp[sn,t]],
                    s=60, c=ACCENT_TEAL, zorder=5, edgecolors="white",
                    linewidths=0.8)

        for ev_idx, ev in enumerate(eng.EVENTS):
            ax2.axvspan(ev['t_start'], ev['t_end'],
                        alpha=0.06, color=EVENT_COLS[ev_idx])

        ax2.set_ylabel("Temp °C", fontsize=7, color=FG_SECONDARY)
        ax2.set_xlabel("Time step", fontsize=7, color=FG_SECONDARY)
        ax2.set_xlim(0, eng.T_STEPS-1)

        self.canvas_ts.draw_idle()

    # ── HEATMAP STRIP ─────────────────────────────────────────────────

    def _draw_heatmap(self, t):
        ax = self.ax_heat
        ax.clear(); ax.set_facecolor(BG_PANEL); self._style_ax(ax)

        im = ax.imshow(self.engine.gms, aspect="auto",
                       cmap=GMS_CMAP, origin="lower",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.axvline(t, color="white", lw=1.5, alpha=0.7)
        ax.axvline(t, color=ACCENT_BLUE, lw=0.7, alpha=0.9)

        ax.set_yticks(range(self.engine.N_NODES))
        ax.set_yticklabels([f"N{i}" for i in range(self.engine.N_NODES)],
                           fontsize=5.5)
        ax.set_xlabel("Time step", fontsize=6)
        ax.set_title("All nodes × time", color=FG_SECONDARY,
                     fontsize=7, fontfamily="monospace", pad=3)

        self.canvas_heat.draw_idle()

    # ── METRICS ──────────────────────────────────────────────────────

    def _update_metrics(self, t):
        eng = self.engine
        sn  = self.selected_node
        gms_val  = eng.gms[sn, t]
        grad_val = eng.grad[sn, t]
        mom_val  = eng.mom[sn, t]
        nis_val  = eng.nis[sn, t]
        dur_val  = eng.dur[sn, t]
        tmp_val  = eng.Temp[sn, t]
        lbl      = eng.label[sn, t]

        self.metric_cards["gms"].config( text=f"{gms_val:.3f}")
        self.metric_cards["grad"].config(text=f"{grad_val:+.2f}°")
        self.metric_cards["mom"].config( text=f"{mom_val:+.2f}°")
        self.metric_cards["nis"].config( text=f"{nis_val:.3f}")
        self.metric_cards["dur"].config( text=f"{dur_val:.3f}")
        self.metric_cards["temp"].config(text=f"{tmp_val:.1f}°")

        if lbl == 2:
            self.status_badge.config(text="▲ HIGH UNSTABLE", fg=ACCENT_RED)
        elif lbl == 1:
            self.status_badge.config(text="◆ MOD. UNSTABLE", fg=ACCENT_AMBER)
        else:
            self.status_badge.config(text="● STABLE", fg=ACCENT_TEAL)

        self.lbl_node.config(text=f"Inspecting: N{sn}  @  t={t}")

    def _update_alerts(self, t):
        eng = self.engine
        txt = self.alert_text
        txt.config(state="normal")

        # Detect transitions at this timestep
        for i in range(eng.N_NODES):
            lbl_now  = eng.label[i, t]
            lbl_prev = eng.label[i, max(0, t-1)]
            if lbl_now != lbl_prev:
                ts_str = f"t={t:03d}"
                if lbl_now == 2:
                    msg = f"[{ts_str}] ▲ N{i} → HIGH UNSTABLE  GMS={eng.gms[i,t]:.2f}\n"
                    txt.insert("1.0", msg, "high")
                elif lbl_now == 1:
                    msg = f"[{ts_str}] ◆ N{i} → MOD UNSTABLE   GMS={eng.gms[i,t]:.2f}\n"
                    txt.insert("1.0", msg, "mod")
                elif lbl_now == 0 and lbl_prev > 0:
                    msg = f"[{ts_str}] ✓ N{i} → STABLE\n"
                    txt.insert("1.0", msg, "ok")

        # Trim log to 60 lines
        lines = int(txt.index("end").split(".")[0])
        if lines > 60:
            txt.delete(f"{lines-10}.0", "end")

        txt.config(state="disabled")

    def _update_performance(self):
        eng = self.engine
        from evaluation.metrics import build_ground_truth

        try:
            sys.path.insert(0, os.path.dirname(__file__))
            gt   = build_ground_truth(eng.N_NODES, eng.T_STEPS)
        except Exception:
            # Inline fallback
            gt = np.zeros((eng.N_NODES, eng.T_STEPS), dtype=int)
            for ev in eng.EVENTS:
                for i in ev['nodes']:
                    gt[i, ev['t_start']:ev['t_end']] = 1

        pred = (eng.gms >= eng.ALPHA).astype(int)
        bl   = eng.baseline

        def metrics(p):
            TP = int(((p==1)&(gt==1)).sum())
            TN = int(((p==0)&(gt==0)).sum())
            FP = int(((p==1)&(gt==0)).sum())
            FN = int(((p==0)&(gt==1)).sum())
            tot = TP+TN+FP+FN
            acc = (TP+TN)/tot  if tot  else 0
            prec= TP/(TP+FP)   if TP+FP else 0
            rec = TP/(TP+FN)   if TP+FN else 0
            far = FP/(FP+TN)   if FP+TN else 0
            return acc, prec, rec, far

        ga, gp, gr, gf = metrics(pred)
        self.perf_labels["Accuracy"].config( text=f"GMS {ga*100:.0f}%")
        self.perf_labels["Precision"].config(text=f"GMS {gp*100:.0f}%")
        self.perf_labels["Recall"].config(   text=f"GMS {gr*100:.0f}%")
        self.perf_labels["FAR"].config(      text=f"GMS {gf*100:.0f}%")

    # ══════════════════════════════════════════════════════════════════
    #  CONTROLS
    # ══════════════════════════════════════════════════════════════════

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.config(text="⏸  PAUSE")
            self._tick()
        else:
            self.btn_play.config(text="▶  PLAY")
            if self._anim_job:
                self.root.after_cancel(self._anim_job)

    def _tick(self):
        if not self.playing:
            return
        if self.t < self.engine.T_STEPS - 1:
            self.t += 1
        else:
            self.t = 0   # loop
        self._draw_all(self.t)
        delay = max(30, int(self.speed_var.get()))
        self._anim_job = self.root.after(delay, self._tick)

    def _step_back(self):
        self.playing = False
        self.btn_play.config(text="▶  PLAY")
        self.t = max(0, self.t - 1)
        self._draw_all(self.t)

    def _step_fwd(self):
        self.playing = False
        self.btn_play.config(text="▶  PLAY")
        self.t = min(self.engine.T_STEPS - 1, self.t + 1)
        self._draw_all(self.t)

    def _reset(self):
        self.playing = False
        self.btn_play.config(text="▶  PLAY")
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
        self.t = 0
        self._draw_all(0)

    def _jump_to(self, t):
        self.t = t
        self._draw_all(t)

    def _on_slider(self, val):
        self.t = int(float(val))
        self._draw_all(self.t)

    def _select_node(self, node_id):
        self.selected_node = node_id
        self._draw_all(self.t)

    def _on_map_click(self, event):
        """Click on sensor map to select nearest node."""
        if event.inaxes != self.ax_map:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        dists = [math.hypot(self.engine.pos[i,0]-x, self.engine.pos[i,1]-y)
                 for i in range(self.engine.N_NODES)]
        nearest = int(np.argmin(dists))
        self._select_node(nearest)

    def _on_weight(self, key):
        """Recompute GMS when any weight changes."""
        w1 = self.w_vars["W1"].get()
        w2 = self.w_vars["W2"].get()
        w3 = self.w_vars["W3"].get()
        w4 = self.w_vars["W4"].get()
        theta  = self.th_vars["THETA"].get()
        window = 8
        alpha  = self.th_vars["ALPHA"].get()
        beta   = self.th_vars["BETA"].get()
        self.engine.rerun(w1, w2, w3, w4, theta, window, alpha, beta)
        self._draw_all(self.t)
        self._update_performance()

    def _on_threshold(self, key):
        self._on_weight(key)   # same recompute path

    def _save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
            initialfile=f"gms_t{self.t:03d}.png"
        )
        if path:
            # Composite: save the map figure
            self.fig_map.savefig(path, dpi=150,
                                  bbox_inches="tight", facecolor=BG_DARK)
            self._log_alert(f"Saved → {os.path.basename(path)}", "info")

    def _log_alert(self, msg, tag="info"):
        txt = self.alert_text
        txt.config(state="normal")
        txt.insert("1.0", f"[SYS] {msg}\n", tag)
        txt.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    # Style ttk widgets to match dark theme
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TScale",
                    background=BG_PANEL,
                    troughcolor=BG_CARD,
                    sliderthickness=14,
                    sliderrelief="flat")
    style.configure("Horizontal.TScale",
                    background=BG_PANEL,
                    troughcolor=BG_CARD)

    app = MissionControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()