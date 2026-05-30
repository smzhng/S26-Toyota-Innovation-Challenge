import queue
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
import math
import threading
import time

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon, FancyArrowPatch
from itertools import cycle
from messages import (
    PauseMessage, ResumeMessage, StopMessage, ToggleGripperMessage,
    PathAssignmentMessage, Waypoint, MotionSettings,
)


# Mission state colours
MISSION_COLORS = {
    "idle":         "#888888",
    "searching":    "#f0a500",
    "target_found": "#2196F3",
    "retrieving":   "#e53935",
    "returning":    "#9c27b0",
    "complete":     "#43a047",
}

MISSION_LABELS = {
    "idle":         "Idle — waiting for mission",
    "searching":    "Searching for target...",
    "target_found": "Target located! Dispatching robots...",
    "retrieving":   "Retrieving target",
    "returning":    "Target acquired! Returning to home...",
    "complete":     "Mission complete",
}


class TelemetryGUI:
    """
    Thread-safe telemetry dashboard.

    Server threads call:
        gui.update_robot(robot_id, telemetry_dict)
        gui.update_mission_state(state, target_x_cm, target_y_cm)
    """

    MAX_VALID_ULTRASONIC_CM = 200.0
    MIN_VALID_ULTRASONIC_CM = 1.0

    FRONT_SENSOR_FORWARD_OFFSET_CM = 9.5
    FRONT_SENSOR_LATERAL_OFFSET_CM = 0.0
    FRONT_SENSOR_ANGLE_OFFSET_DEG = 0.0

    LEFT_SENSOR_FORWARD_OFFSET_CM = 0.0
    LEFT_SENSOR_LATERAL_OFFSET_CM = 16.0
    LEFT_SENSOR_ANGLE_OFFSET_DEG = 90.0

    ARENA_SIZE_CM = 400
    GRID_SPACING_CM = 10
    ECHO_WINDOW_S = 10.0
    GRID_DIM_CELLS = 40
    DEFAULT_TEST_DISTANCE_CM = 30.0
    DEFAULT_TURN_SPEED = 150
    DEFAULT_DRIVE_SPEED = 200
    SAFETY_BOX_SIZE_CM = 40.0
    SAFETY_BOX_HALF_CM = SAFETY_BOX_SIZE_CM / 2.0

    def __init__(self, command_sender=None):
        self.root = tk.Tk()
        self.root.title("Fleet Mission Dashboard")
        self.root.geometry("1600x900")

        self.command_sender = command_sender
        self.telemetry_queue = queue.Queue()
        self.mission_queue = queue.Queue()
        self.test_path_counter = int(time.time())

        self.robot_states = {}
        self.robot_history = defaultdict(lambda: {
            "t": [], "x": [], "y": [], "theta": [],
            "front_ultra": [], "left_ultra": [],
            "front_echo_t": [], "front_echo_x": [], "front_echo_y": [],
            "left_echo_t": [], "left_echo_x": [], "left_echo_y": [],
        })

        # Planned path overlays: robot_id -> list of (x_cm, y_cm)
        self.planned_paths = {}

        # Occupancy grid map: 40x40 numpy array of probabilities [0..1]
        # 0.5 = unknown, 0.0 = free, 1.0 = occupied
        self._occupancy_map: np.ndarray | None = None
        self._occupancy_lock = threading.Lock()

        # Mission state
        self._mission_state = "idle"
        self._target_x_cm = None
        self._target_y_cm = None

        self.robot_colors = {}
        self.color_cycle = cycle([
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:cyan",
        ])

        self.event_log_queue = queue.Queue()

        self._build_layout()
        self.root.after(100, self._process_queue)

    # ----------------------------
    # Public API
    # ----------------------------
    def update_robot(self, robot_id: str, telemetry: dict):
        self.telemetry_queue.put(("telemetry", robot_id, telemetry))

    def update_occupancy_map(self, probability_map: np.ndarray):
        """Called by arbiter with a fresh 40x40 occupancy probability array."""
        with self._occupancy_lock:
            self._occupancy_map = probability_map.copy()

    def update_mission_state(self, state: str, target_x_cm: float = None, target_y_cm: float = None):
        """Called by the arbiter when mission state changes."""
        self.mission_queue.put((state, target_x_cm, target_y_cm))

    def update_planned_path(self, robot_id: str, path_cells: list):
        """Called by arbiter to push a planned path for overlay drawing."""
        waypoints = [
            (col * 10 + 5, row * 10 + 5)
            for row, col in path_cells
        ]
        self.telemetry_queue.put(("path_overlay", robot_id, waypoints))

    def log_path_event(self, robot_id: str, event_type: str, msg: dict):
        """Called by arbiter when a path lifecycle event arrives from a robot."""
        self.event_log_queue.put((robot_id, event_type, dict(msg)))

    # ----------------------------
    # Layout
    # ----------------------------
    def _build_layout(self):
        mainframe = ttk.Frame(self.root, padding=8)
        mainframe.pack(fill=tk.BOTH, expand=True)

        # Left panel
        left_panel = ttk.Frame(mainframe, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_panel.pack_propagate(False)

        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ── Mission status panel ──────────────────────────────────────────
        mission_frame = ttk.LabelFrame(scrollable_frame, text="Mission Status")
        mission_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(0, 8))

        self._mission_status_var = tk.StringVar(value=MISSION_LABELS["idle"])
        self._mission_status_label = tk.Label(
            mission_frame,
            textvariable=self._mission_status_var,
            font=("Helvetica", 11, "bold"),
            fg=MISSION_COLORS["idle"],
            wraplength=280,
            justify=tk.LEFT,
        )
        self._mission_status_label.pack(fill=tk.X, padx=6, pady=6)

        self._target_coords_var = tk.StringVar(value="Target: not yet detected")
        tk.Label(
            mission_frame,
            textvariable=self._target_coords_var,
            font=("Helvetica", 9),
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        # ── Robot table ───────────────────────────────────────────────────
        robots_frame = ttk.LabelFrame(scrollable_frame, text="Robots")
        robots_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(0, 8))

        columns = ("robot_id", "state", "x", "y", "theta", "dest", "wpts")
        self.tree = ttk.Treeview(robots_frame, columns=columns, show="headings", height=4)
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.column("robot_id", width=70,  anchor="center")
        self.tree.column("state",    width=80,  anchor="center")
        self.tree.column("x",        width=45,  anchor="center")
        self.tree.column("y",        width=45,  anchor="center")
        self.tree.column("theta",    width=45,  anchor="center")
        self.tree.column("dest",     width=90,  anchor="center")
        self.tree.column("wpts",     width=40,  anchor="center")
        self.tree.pack(fill=tk.X, expand=False)

        # ── Arena plot ────────────────────────────────────────────────────
        right_panel = ttk.Frame(mainframe)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.ax_traj = self.fig.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Manual controls ───────────────────────────────────────────────
        controls_frame = ttk.LabelFrame(scrollable_frame, text="Robot Controls")
        controls_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(8, 0))

        ttk.Label(controls_frame, text="Target Robot:").pack(fill=tk.X, pady=(4, 2))
        self.selected_robot_var = tk.StringVar(value="")
        self.robot_selector = ttk.Combobox(
            controls_frame, textvariable=self.selected_robot_var,
            state="readonly", values=[],
        )
        self.robot_selector.pack(fill=tk.X, pady=(0, 4))

        self.selected_robot_summary_var = tk.StringVar(value="Select a robot.")
        ttk.Label(controls_frame, textvariable=self.selected_robot_summary_var,
                  wraplength=280, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 6))

        for label, cmd in [
            ("Pause",             self._send_pause),
            ("Resume",            self._send_resume),
            ("Stop",              self._send_stop),
            ("Toggle Gripper",    self._send_toggle_gripper),
            ("Send Straight Test",self._send_straight_test_path),
            ("Send 180 Turn Test",self._send_turnaround_test_path),
            ("Send L Test Path",  self._send_test_path),
        ]:
            ttk.Button(controls_frame, text=label, command=cmd).pack(fill=tk.X, pady=2)

        # ── Go To Coordinates panel ───────────────────────────────────────
        goto_frame = ttk.LabelFrame(scrollable_frame, text="Go To Coordinates")
        goto_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(8, 0))

        ttk.Label(goto_frame,
                  text="Send selected robot to an exact position.",
                  wraplength=280, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=(4, 4))

        goto_coord_row = ttk.Frame(goto_frame)
        goto_coord_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Label(goto_coord_row, text="X (cm)").grid(row=0, column=0, sticky="w")
        self.goto_x_var = tk.StringVar(value="")
        ttk.Entry(goto_coord_row, textvariable=self.goto_x_var, width=7).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(goto_coord_row, text="Y (cm)").grid(row=0, column=2, sticky="w")
        self.goto_y_var = tk.StringVar(value="")
        ttk.Entry(goto_coord_row, textvariable=self.goto_y_var, width=7).grid(row=0, column=3, padx=(4, 0))

        ttk.Button(goto_frame, text="Send Robot to Coordinates",
                   command=self._send_goto_waypoint).pack(fill=tk.X, padx=4, pady=(0, 6))

        # ── Path Events log ───────────────────────────────────────────────
        event_log_frame = ttk.LabelFrame(scrollable_frame, text="Path Events")
        event_log_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(8, 0))

        log_inner = ttk.Frame(event_log_frame)
        log_inner.pack(fill=tk.X, expand=False)
        self._event_log = tk.Text(log_inner, height=8, font=("Courier", 8),
                                   wrap="word", state="disabled", bg="#1e1e1e", fg="#d4d4d4",
                                   relief="flat")
        log_scrollbar = ttk.Scrollbar(log_inner, command=self._event_log.yview)
        self._event_log.configure(yscrollcommand=log_scrollbar.set)
        self._event_log.pack(side=tk.LEFT, fill=tk.X, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._event_log.tag_configure("started",  foreground="#4ec9b0")
        self._event_log.tag_configure("reached",  foreground="#dcdcaa")
        self._event_log.tag_configure("complete", foreground="#4fc1ff")

        ttk.Button(event_log_frame, text="Clear Log",
                   command=self._clear_event_log).pack(fill=tk.X, padx=4, pady=(2, 4))

        # ── Retrieval mission panel ───────────────────────────────────────
        mission_ctrl_frame = ttk.LabelFrame(scrollable_frame, text="Retrieval Mission")
        mission_ctrl_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(8, 0))

        ttk.Label(mission_ctrl_frame,
                  text="Manually set target position and launch mission.\n"
                       "Vision will override these when it detects the door.",
                  wraplength=280, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=(4, 6))

        coord_row = ttk.Frame(mission_ctrl_frame)
        coord_row.pack(fill=tk.X, padx=4, pady=(0, 6))
        ttk.Label(coord_row, text="X (cm)").grid(row=0, column=0, sticky="w")
        self.mission_x_var = tk.StringVar(value="200")
        ttk.Entry(coord_row, textvariable=self.mission_x_var, width=7).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(coord_row, text="Y (cm)").grid(row=0, column=2, sticky="w")
        self.mission_y_var = tk.StringVar(value="200")
        ttk.Entry(coord_row, textvariable=self.mission_y_var, width=7).grid(row=0, column=3, padx=(4, 0))

        ttk.Button(
            mission_ctrl_frame, text="Start Retrieval Mission",
            command=self._send_retrieval_mission,
        ).pack(fill=tk.X, padx=4, pady=(0, 4))

        ttk.Button(
            mission_ctrl_frame, text="Reset Mission",
            command=self._reset_mission,
        ).pack(fill=tk.X, padx=4, pady=(0, 6))

        # ── Multi-Robot Navigate panel ────────────────────────────────────
        coord_frame = ttk.LabelFrame(scrollable_frame, text="Multi-Robot Navigate")
        coord_frame.pack(fill=tk.X, expand=False, padx=(0, 8), pady=(8, 0))

        ttk.Label(coord_frame,
                  text="Routes both robots simultaneously using space-time A* — "
                       "paths are collision-free by design.",
                  wraplength=280, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=(4, 6))

        self.grid_robot_one_var = tk.StringVar(value="")
        self.grid_robot_two_var = tk.StringVar(value="")

        for label, robot_var, combo_attr, x_attr, y_attr, x_def, y_def in [
            ("Robot 1", self.grid_robot_one_var, "multi_robot_one_combo",
             "multi_robot_one_x_var", "multi_robot_one_y_var", "100", "100"),
            ("Robot 2", self.grid_robot_two_var, "multi_robot_two_combo",
             "multi_robot_two_x_var", "multi_robot_two_y_var", "200", "200"),
        ]:
            ttk.Label(coord_frame, text=label).pack(fill=tk.X, padx=4)
            combo = ttk.Combobox(coord_frame, textvariable=robot_var, state="readonly", values=[])
            combo.pack(fill=tk.X, padx=4, pady=(0, 4))
            setattr(self, combo_attr, combo)
            goal_row = ttk.Frame(coord_frame)
            goal_row.pack(fill=tk.X, padx=4, pady=(0, 6))
            ttk.Label(goal_row, text="Target X (cm)").grid(row=0, column=0, sticky="w")
            setattr(self, x_attr, tk.StringVar(value=x_def))
            ttk.Entry(goal_row, textvariable=getattr(self, x_attr), width=6).grid(row=0, column=1, padx=(4, 12))
            ttk.Label(goal_row, text="Y (cm)").grid(row=0, column=2, sticky="w")
            setattr(self, y_attr, tk.StringVar(value=y_def))
            ttk.Entry(goal_row, textvariable=getattr(self, y_attr), width=6).grid(row=0, column=3, padx=(4, 0))

        self.grid_plan_summary_var = tk.StringVar(value="Select two robots and set target positions.")
        ttk.Label(coord_frame, textvariable=self.grid_plan_summary_var,
                  wraplength=280, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=(0, 6))
        ttk.Button(coord_frame, text="Navigate Both Robots",
                   command=self._send_two_robot_traverse).pack(fill=tk.X, padx=4, pady=(0, 6))

    # ----------------------------
    # Helpers
    # ----------------------------
    def _get_robot_color(self, robot_id: str) -> str:
        if robot_id not in self.robot_colors:
            self.robot_colors[robot_id] = next(self.color_cycle)
        return self.robot_colors[robot_id]

    def _draw_robot_safety_box(self, ax, x, y, theta_deg, label, color):
        half = self.SAFETY_BOX_HALF_CM
        theta_rad = math.radians(theta_deg)
        local = [(-half, -half), (half, -half), (half, half), (-half, half)]
        world = [
            (x + lx * math.cos(theta_rad) - ly * math.sin(theta_rad),
             y + lx * math.sin(theta_rad) + ly * math.cos(theta_rad))
            for lx, ly in local
        ]
        ax.add_patch(Polygon(world, closed=True, fill=False,
                             linewidth=1.5, linestyle="-", alpha=0.8,
                             label=label, edgecolor=color))

    def _world_sensor_position(self, x, y, theta_deg, fwd, lat):
        theta_rad = math.radians(theta_deg)
        sx = x + fwd * math.cos(theta_rad) - lat * math.sin(theta_rad)
        sy = y + fwd * math.sin(theta_rad) + lat * math.cos(theta_rad)
        return sx, sy

    def _compute_echo_point(self, x, y, theta_deg, dist_cm, fwd, lat, angle_offset):
        if not (self.MIN_VALID_ULTRASONIC_CM <= dist_cm <= self.MAX_VALID_ULTRASONIC_CM):
            return None
        sx, sy = self._world_sensor_position(x, y, theta_deg, fwd, lat)
        ray_rad = math.radians(theta_deg + angle_offset)
        return sx + dist_cm * math.cos(ray_rad), sy + dist_cm * math.sin(ray_rad)

    def _draw_latest_ray(self, ax, x, y, theta_deg, dist_cm, fwd, lat, angle_offset, label, linestyle, color):
        if not (self.MIN_VALID_ULTRASONIC_CM <= dist_cm <= self.MAX_VALID_ULTRASONIC_CM):
            return
        sx, sy = self._world_sensor_position(x, y, theta_deg, fwd, lat)
        ray_rad = math.radians(theta_deg + angle_offset)
        ex = sx + dist_cm * math.cos(ray_rad)
        ey = sy + dist_cm * math.sin(ray_rad)
        ax.plot([sx, ex], [sy, ey], linestyle=linestyle, label=label, color=color, alpha=0.6)

    def _prune_echo_history(self, hist, current_t_s):
        cutoff = current_t_s - self.ECHO_WINDOW_S
        for key_t, key_x, key_y in [
            ("front_echo_t", "front_echo_x", "front_echo_y"),
            ("left_echo_t",  "left_echo_x",  "left_echo_y"),
        ]:
            while hist[key_t] and hist[key_t][0] < cutoff:
                hist[key_t].pop(0)
                hist[key_x].pop(0)
                hist[key_y].pop(0)

    # ----------------------------
    # Queue processing
    # ----------------------------
    def _process_queue(self):
        updated = False

        # Mission state updates
        while not self.mission_queue.empty():
            state, tx, ty = self.mission_queue.get()
            self._mission_state = state
            if tx is not None:
                self._target_x_cm = tx
            if ty is not None:
                self._target_y_cm = ty
            self._mission_status_var.set(MISSION_LABELS.get(state, state))
            self._mission_status_label.config(fg=MISSION_COLORS.get(state, "#888888"))
            if self._target_x_cm is not None:
                self._target_coords_var.set(
                    f"Target: ({self._target_x_cm:.0f}, {self._target_y_cm:.0f}) cm"
                )
            updated = True

        # Telemetry and path overlay updates
        while not self.telemetry_queue.empty():
            item = self.telemetry_queue.get()

            if item[0] == "path_overlay":
                _, robot_id, waypoints = item
                self.planned_paths[robot_id] = waypoints
                updated = True
                continue

            _, robot_id, telemetry = item
            self.robot_states[robot_id] = telemetry

            # Update planned path from arbiter snapshot if present
            if "planned_path_cells" in telemetry:
                cells = telemetry["planned_path_cells"]
                if cells:
                    self.planned_paths[robot_id] = [
                        (col * 10 + 5, row * 10 + 5) for row, col in cells
                    ]

            hist = self.robot_history[robot_id]
            t    = telemetry.get("t_ms")
            x    = telemetry.get("x_cm")
            y    = telemetry.get("y_cm")
            theta = telemetry.get("theta_deg")
            front = telemetry.get("front_ultrasonic_cm")
            left  = telemetry.get("left_ultrasonic_cm")
            t_s   = float(t) / 1000.0 if t is not None else None

            if t_s is not None: hist["t"].append(t_s)
            if x    is not None: hist["x"].append(float(x))
            if y    is not None: hist["y"].append(float(y))
            if theta is not None: hist["theta"].append(float(theta))
            if front is not None: hist["front_ultra"].append(float(front))
            if left  is not None: hist["left_ultra"].append(float(left))

            if x is not None and y is not None and theta is not None:
                fx, fy = float(x), float(y)
                ft = float(theta)
                if front is not None:
                    pt = self._compute_echo_point(fx, fy, ft, float(front),
                         self.FRONT_SENSOR_FORWARD_OFFSET_CM, self.FRONT_SENSOR_LATERAL_OFFSET_CM,
                         self.FRONT_SENSOR_ANGLE_OFFSET_DEG)
                    if pt:
                        hist["front_echo_t"].append(t_s or 0.0)
                        hist["front_echo_x"].append(pt[0])
                        hist["front_echo_y"].append(pt[1])
                if left is not None:
                    pt = self._compute_echo_point(fx, fy, ft, float(left),
                         self.LEFT_SENSOR_FORWARD_OFFSET_CM, self.LEFT_SENSOR_LATERAL_OFFSET_CM,
                         self.LEFT_SENSOR_ANGLE_OFFSET_DEG)
                    if pt:
                        hist["left_echo_t"].append(t_s or 0.0)
                        hist["left_echo_x"].append(pt[0])
                        hist["left_echo_y"].append(pt[1])

            if t_s is not None:
                self._prune_echo_history(hist, t_s)

            updated = True

        # Event log updates (drain synchronously — no redraw needed)
        while not self.event_log_queue.empty():
            robot_id, event_type, msg = self.event_log_queue.get()
            self._append_event_log(robot_id, event_type, msg)

        if updated:
            self._refresh_table()
            self._refresh_robot_selector()
            self._refresh_plot()

        self.root.after(100, self._process_queue)

    # ----------------------------
    # Plot
    # ----------------------------
    def _refresh_plot(self):
        self.ax_traj.clear()

        self.ax_traj.set_title("Fleet Arena — Live View", fontsize=12)
        self.ax_traj.set_xlabel("x (cm)")
        self.ax_traj.set_ylabel("y (cm)")
        self.ax_traj.set_xlim(0, self.ARENA_SIZE_CM)
        self.ax_traj.set_ylim(0, self.ARENA_SIZE_CM)
        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.xaxis.set_major_locator(MultipleLocator(50))
        self.ax_traj.yaxis.set_major_locator(MultipleLocator(50))
        self.ax_traj.xaxis.set_minor_locator(MultipleLocator(10))
        self.ax_traj.yaxis.set_minor_locator(MultipleLocator(10))
        self.ax_traj.grid(which="major", linewidth=1.0)
        self.ax_traj.grid(which="minor", linewidth=0.3)

        # ── Occupancy map heatmap ─────────────────────────────────────────
        # Drawn first so it sits behind everything else.
        # Cells: 0.0=free(white), 0.5=unknown(grey), 1.0=occupied(dark)
        with self._occupancy_lock:
            omap = self._occupancy_map.copy() if self._occupancy_map is not None else None

        if omap is not None:
            # Flip vertically: row 0 is y=0 (bottom of arena in our coords)
            display_map = np.flipud(omap)
            self.ax_traj.imshow(
                display_map,
                extent=[0, self.ARENA_SIZE_CM, 0, self.ARENA_SIZE_CM],
                origin="upper",
                cmap="RdYlGn_r",   # green=free, yellow=unknown, red=occupied
                vmin=0.0, vmax=1.0,
                alpha=0.45,
                zorder=0,
                interpolation="nearest",
            )

        # ── Target marker ─────────────────────────────────────────────────
        if self._target_x_cm is not None and self._target_y_cm is not None:
            tx, ty = self._target_x_cm, self._target_y_cm
            color = MISSION_COLORS.get(self._mission_state, "red")

            # Pulsing X marker
            self.ax_traj.plot(tx, ty, "x", color=color,
                              markersize=18, markeredgewidth=3, zorder=10,
                              label="Target (car door)")

            # Dashed circle around target
            circle = plt_circle = __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                (tx, ty), 15, fill=False, linestyle="--",
                edgecolor=color, linewidth=1.5, alpha=0.7, zorder=9,
            )
            self.ax_traj.add_patch(circle)

            self.ax_traj.annotate(
                f"Target\n({tx:.0f}, {ty:.0f})",
                xy=(tx, ty), xytext=(tx + 20, ty + 20),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )

        # ── Staging position marker ───────────────────────────────────────
        # Import here to keep top-level clean
        staging_x = 50.0
        staging_y = 50.0
        self.ax_traj.plot(staging_x, staging_y, "s",
                          color="tab:gray", markersize=10, alpha=0.5,
                          label="Robot B staging", zorder=5)

        # ── Planned path overlays ─────────────────────────────────────────
        for robot_id, path_waypoints in self.planned_paths.items():
            if len(path_waypoints) < 2:
                continue
            color = self._get_robot_color(robot_id)
            xs = [p[0] for p in path_waypoints]
            ys = [p[1] for p in path_waypoints]
            self.ax_traj.plot(xs, ys,
                              linestyle="--", linewidth=1.5,
                              color=color, alpha=0.45,
                              label=f"{robot_id} planned path", zorder=4)
            # Arrow at midpoint to show direction
            mid = len(xs) // 2
            if mid > 0:
                dx = xs[mid] - xs[mid - 1]
                dy = ys[mid] - ys[mid - 1]
                self.ax_traj.annotate(
                    "", xy=(xs[mid], ys[mid]),
                    xytext=(xs[mid] - dx * 0.001, ys[mid] - dy * 0.001),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, alpha=0.6),
                )

        # ── Robot trajectories ────────────────────────────────────────────
        for robot_id, hist in self.robot_history.items():
            xs = hist["x"]
            ys = hist["y"]
            color = self._get_robot_color(robot_id)

            if xs and ys:
                self.ax_traj.plot(xs, ys, marker=".", markersize=3,
                                  linestyle="-", alpha=0.4,
                                  label=f"{robot_id} trail", color=color)

                theta_deg = hist["theta"][-1] if hist["theta"] else 0.0
                theta_rad = math.radians(theta_deg)
                arrow_len = 10.0

                # Robot body dot
                self.ax_traj.scatter([xs[-1]], [ys[-1]], s=80,
                                     label=f"{robot_id}", zorder=6, color=color)

                # Heading arrow
                self.ax_traj.arrow(
                    xs[-1], ys[-1],
                    arrow_len * math.cos(theta_rad),
                    arrow_len * math.sin(theta_rad),
                    head_width=4.0, head_length=5.0,
                    length_includes_head=True,
                    color=color, zorder=7,
                )

                # Safety box
                self._draw_robot_safety_box(self.ax_traj, xs[-1], ys[-1], theta_deg,
                                            label=f"{robot_id} safety box", color=color)

                # Ultrasonic rays
                if hist["front_ultra"]:
                    self._draw_latest_ray(
                        self.ax_traj, xs[-1], ys[-1], theta_deg, hist["front_ultra"][-1],
                        self.FRONT_SENSOR_FORWARD_OFFSET_CM, self.FRONT_SENSOR_LATERAL_OFFSET_CM,
                        self.FRONT_SENSOR_ANGLE_OFFSET_DEG,
                        f"{robot_id} front sonar", "--", color,
                    )

            # Ultrasonic echo scatter points
            for ex_key, ey_key, elabel in [
                ("front_echo_x", "front_echo_y", f"{robot_id} front hits"),
                ("left_echo_x",  "left_echo_y",  f"{robot_id} left hits"),
            ]:
                if hist[ex_key] and hist[ey_key]:
                    self.ax_traj.scatter(hist[ex_key], hist[ey_key],
                                         s=15, alpha=0.6, label=elabel, color=color)

        # ── Mission state banner ──────────────────────────────────────────
        banner_color = MISSION_COLORS.get(self._mission_state, "#888888")
        banner_text  = MISSION_LABELS.get(self._mission_state, self._mission_state)
        self.ax_traj.text(
            0.5, 0.98, banner_text,
            transform=self.ax_traj.transAxes,
            fontsize=10, fontweight="bold",
            color=banner_color, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=banner_color),
        )

        handles, labels = self.ax_traj.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            self.ax_traj.legend(unique.values(), unique.keys(),
                                loc="lower right", fontsize=7)

        self.canvas_plot.draw()

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for robot_id, state in self.robot_states.items():
            dest = state.get("task_destination")
            dest_str = f"({dest['x_cm']:.0f},{dest['y_cm']:.0f})" if dest else "—"
            wpts = state.get("waypoints_remaining", 0)
            wpts_str = str(wpts) if wpts else "—"
            self.tree.insert("", "end", values=(
                robot_id,
                state.get("state", ""),
                round(float(state.get("x_cm", 0.0)), 1),
                round(float(state.get("y_cm", 0.0)), 1),
                round(float(state.get("theta_deg", 0.0)), 1),
                dest_str,
                wpts_str,
            ))

    def _refresh_robot_selector(self):
        robot_ids = sorted(self.robot_states.keys())
        for combo_var, combo_name in [
            (self.selected_robot_var, "robot_selector"),
            (self.grid_robot_one_var, "multi_robot_one_combo"),
            (self.grid_robot_two_var, "multi_robot_two_combo"),
        ]:
            widget = getattr(self, combo_name, None)
            if widget:
                widget["values"] = robot_ids
            if combo_var.get() not in robot_ids and robot_ids:
                combo_var.set(robot_ids[0])

        if not self.grid_robot_two_var.get() and len(robot_ids) > 1:
            self.grid_robot_two_var.set(robot_ids[1])

        selected_state = self.robot_states.get(self.selected_robot_var.get())
        if selected_state:
            robot_id = self.selected_robot_var.get()
            dest = selected_state.get("task_destination")
            wpts = selected_state.get("waypoints_remaining", 0)
            dest_str = f" → ({dest['x_cm']:.0f},{dest['y_cm']:.0f}) [{wpts} left]" if dest else ""
            self.selected_robot_summary_var.set(
                f"{robot_id}: {selected_state.get('state','?')}  "
                f"({float(selected_state.get('x_cm',0)):.0f}, "
                f"{float(selected_state.get('y_cm',0)):.0f}) cm  "
                f"{float(selected_state.get('theta_deg',0)):.0f}°"
                f"{dest_str}"
            )
        else:
            self.selected_robot_summary_var.set("Select a robot.")

    # ----------------------------
    # Command senders
    # ----------------------------
    def _get_selected_robot_id(self):
        v = self.selected_robot_var.get().strip()
        return v if v else None

    def _get_selected_robot_state(self):
        rid = self._get_selected_robot_id()
        return self.robot_states.get(rid) if rid else None

    def _get_motion_settings(self):
        return MotionSettings(
            turn_speed_deg_per_sec=self.DEFAULT_TURN_SPEED,
            drive_speed_deg_per_sec=self.DEFAULT_DRIVE_SPEED,
        )

    def _next_test_path_id(self):
        self.test_path_counter += 1
        return self.test_path_counter

    def _send_pause(self):
        rid = self._get_selected_robot_id()
        if rid and self.command_sender:
            self.command_sender(PauseMessage(robot_id=rid, reason="gui_pause_button"))

    def _send_resume(self):
        rid = self._get_selected_robot_id()
        if rid and self.command_sender:
            self.command_sender(ResumeMessage(robot_id=rid))

    def _send_stop(self):
        rid = self._get_selected_robot_id()
        if rid and self.command_sender:
            self.command_sender(StopMessage(robot_id=rid, reason="gui_stop_button"))

    def _send_toggle_gripper(self):
        rid = self._get_selected_robot_id()
        if rid and self.command_sender:
            self.command_sender(ToggleGripperMessage(robot_id=rid))

    def _send_straight_test_path(self):
        rid = self._get_selected_robot_id()
        state = self._get_selected_robot_state()
        if not rid or not state or not self.command_sender:
            return
        x = float(state.get("x_cm", 0.0))
        y = float(state.get("y_cm", 0.0))
        theta_rad = math.radians(float(state.get("theta_deg", 0.0)))
        d = self.DEFAULT_TEST_DISTANCE_CM
        self.command_sender(PathAssignmentMessage(
            robot_id=rid, path_id=self._next_test_path_id(), replace_existing=True,
            waypoints=[Waypoint(x_cm=min(max(x + d*math.cos(theta_rad), 0), 400),
                                y_cm=min(max(y + d*math.sin(theta_rad), 0), 400))],
            motion=self._get_motion_settings(),
        ))

    def _send_turnaround_test_path(self):
        rid = self._get_selected_robot_id()
        state = self._get_selected_robot_state()
        if not rid or not state or not self.command_sender:
            return
        x = float(state.get("x_cm", 0.0))
        y = float(state.get("y_cm", 0.0))
        theta_rad = math.radians(float(state.get("theta_deg", 0.0)))
        d = self.DEFAULT_TEST_DISTANCE_CM
        self.command_sender(PathAssignmentMessage(
            robot_id=rid, path_id=self._next_test_path_id(), replace_existing=True,
            waypoints=[Waypoint(x_cm=min(max(x - d*math.cos(theta_rad), 0), 400),
                                y_cm=min(max(y - d*math.sin(theta_rad), 0), 400))],
            motion=self._get_motion_settings(),
        ))

    def _send_test_path(self):
        rid = self._get_selected_robot_id()
        state = self._get_selected_robot_state()
        if not rid or not state or not self.command_sender:
            return
        x = float(state.get("x_cm", 0.0))
        y = float(state.get("y_cm", 0.0))
        self.command_sender(PathAssignmentMessage(
            robot_id=rid, path_id=self._next_test_path_id(), replace_existing=True,
            waypoints=[
                Waypoint(x_cm=min(max(x+20, 0), 400), y_cm=min(max(y,    0), 400)),
                Waypoint(x_cm=min(max(x+40, 0), 400), y_cm=min(max(y,    0), 400)),
                Waypoint(x_cm=min(max(x+40, 0), 400), y_cm=min(max(y+40, 0), 400)),
            ],
            motion=self._get_motion_settings(),
        ))

    def _send_retrieval_mission(self):
        if not self.command_sender:
            return
        try:
            x_cm = float(self.mission_x_var.get())
            y_cm = float(self.mission_y_var.get())
        except ValueError:
            self._mission_status_var.set("Invalid coordinates — enter numbers only")
            return

        self._mission_state = "retrieving"
        self._target_x_cm = x_cm
        self._target_y_cm = y_cm
        self._mission_status_var.set(MISSION_LABELS["retrieving"])
        self._mission_status_label.config(fg=MISSION_COLORS["retrieving"])
        self._target_coords_var.set(f"Target: ({x_cm:.0f}, {y_cm:.0f}) cm")

        self.command_sender({
            "type": "retrieval_mission",
            "x_cm": x_cm,
            "y_cm": y_cm,
        })

    def _reset_mission(self):
        self._mission_state = "idle"
        self._target_x_cm = None
        self._target_y_cm = None
        self.planned_paths.clear()
        self._mission_status_var.set(MISSION_LABELS["idle"])
        self._mission_status_label.config(fg=MISSION_COLORS["idle"])
        self._target_coords_var.set("Target: not yet detected")

    def _parse_goal_cm(self, x_var, y_var):
        try:
            x = float(x_var.get())
            y = float(y_var.get())
        except (TypeError, ValueError):
            return None
        col = max(0, min(self.GRID_DIM_CELLS - 1, int(x / 10)))
        row = max(0, min(self.GRID_DIM_CELLS - 1, int(y / 10)))
        return row, col

    def _send_two_robot_traverse(self):
        if not self.command_sender:
            return
        r1 = self.grid_robot_one_var.get().strip()
        r2 = self.grid_robot_two_var.get().strip()
        if not r1 or not r2 or r1 == r2:
            self.grid_plan_summary_var.set("Pick two different robots.")
            return
        g1 = self._parse_goal_cm(self.multi_robot_one_x_var, self.multi_robot_one_y_var)
        g2 = self._parse_goal_cm(self.multi_robot_two_x_var, self.multi_robot_two_y_var)
        if g1 is None or g2 is None:
            self.grid_plan_summary_var.set("Enter valid X/Y coordinates (cm).")
            return
        if g1 == g2:
            self.grid_plan_summary_var.set("Targets map to the same grid cell — choose farther apart positions.")
            return
        x1 = float(self.multi_robot_one_x_var.get())
        y1 = float(self.multi_robot_one_y_var.get())
        x2 = float(self.multi_robot_two_x_var.get())
        y2 = float(self.multi_robot_two_y_var.get())
        self.grid_plan_summary_var.set(
            f"Planning: {r1}→({x1:.0f},{y1:.0f}), {r2}→({x2:.0f},{y2:.0f})"
        )
        self.command_sender({
            "type": "coordinated_traverse",
            "robots": [
                {"robot_id": r1, "goal_row": g1[0], "goal_col": g1[1]},
                {"robot_id": r2, "goal_row": g2[0], "goal_col": g2[1]},
            ],
        })

    def _send_goto_waypoint(self):
        rid = self._get_selected_robot_id()
        if not rid or not self.command_sender:
            return
        try:
            x_cm = float(self.goto_x_var.get())
            y_cm = float(self.goto_y_var.get())
        except ValueError:
            return
        x_cm = min(max(x_cm, 0.0), 400.0)
        y_cm = min(max(y_cm, 0.0), 400.0)
        motion = self._get_motion_settings()
        self.command_sender({
            "type": "goto_with_avoidance",
            "robot_id": rid,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "motion": {
                "turn_speed_deg_per_sec": motion.turn_speed_deg_per_sec,
                "drive_speed_deg_per_sec": motion.drive_speed_deg_per_sec,
            },
        })

    def _append_event_log(self, robot_id: str, event_type: str, msg: dict):
        import time as _time
        ts = _time.strftime("%H:%M:%S")

        if event_type == "path_started":
            line = f"[{ts}] {robot_id}  PATH STARTED  path={msg.get('path_id', '?')}\n"
            tag = "started"
        elif event_type == "waypoint_reached":
            x = msg.get("x_cm")
            y = msg.get("y_cm")
            idx = msg.get("waypoint_index", "?")
            coord = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "?"
            line = f"[{ts}] {robot_id}  WAYPOINT #{idx} reached  {coord}\n"
            tag = "reached"
        elif event_type == "path_complete":
            x = msg.get("x_cm")
            y = msg.get("y_cm")
            coord = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "?"
            line = f"[{ts}] {robot_id}  PATH COMPLETE  {coord}\n"
            tag = "complete"
        else:
            line = f"[{ts}] {robot_id}  {event_type}\n"
            tag = None

        self._event_log.configure(state="normal")
        if tag:
            self._event_log.insert("end", line, tag)
        else:
            self._event_log.insert("end", line)
        self._event_log.see("end")
        self._event_log.configure(state="disabled")

    def _clear_event_log(self):
        self._event_log.configure(state="normal")
        self._event_log.delete("1.0", "end")
        self._event_log.configure(state="disabled")

    def run(self):
        self.root.mainloop()