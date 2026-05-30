import json
from collections import deque
import heapq
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from gui import TelemetryGUI

HOST = "0.0.0.0"
PORT = 9000
MAX_CLIENTS = 10
GRID_CELL_CM = 10.0
GRID_DIM_CELLS = 40

# Staging position for robot B during retrieval missions (centre of arena, near edge)
# Tune this to a safe corner of your physical arena
STAGING_X_CM = 50.0
STAGING_Y_CM = 50.0


# ----------------------------
# Occupancy Grid
# ----------------------------
# Each cell is a float in [0.0, 1.0]:
#   0.0 = certainly free
#   0.5 = unknown (initial state)
#   1.0 = certainly occupied
#
# We use a log-odds update so repeated sensor hits accumulate confidence
# and single noise hits don't instantly mark a cell as occupied.

CELL_SIZE_CM = GRID_CELL_CM          # 10 cm per cell
P_OCCUPIED_HIT  = 0.85               # probability obstacle is there given sensor hit
P_OCCUPIED_FREE = 0.15               # probability obstacle is there given sensor ray passed through
LOG_ODDS_MAX =  5.0                  # clamp to avoid overconfidence
LOG_ODDS_MIN = -5.0
MAX_ULTRASONIC_CM = 180.0            # ignore readings beyond this (sensor noise floor)
MIN_ULTRASONIC_CM = 3.0


def _log_odds(p: float) -> float:
    return math.log(p / (1.0 - p))


LOG_ODDS_HIT  = _log_odds(P_OCCUPIED_HIT)
LOG_ODDS_FREE = _log_odds(P_OCCUPIED_FREE)
LOG_ODDS_PRIOR = 0.0   # log-odds of p=0.5


class OccupancyGrid:
    """
    40x40 log-odds occupancy grid updated from ultrasonic telemetry.
    Thread-safe: all public methods acquire self._lock.
    """

    def __init__(self, dim: int = GRID_DIM_CELLS):
        self.dim = dim
        self._lock = threading.Lock()
        # log-odds grid, initialised to 0 (unknown, p=0.5)
        self._grid = np.zeros((dim, dim), dtype=np.float32)

    def _cm_to_cell(self, x_cm: float, y_cm: float) -> tuple[int, int] | None:
        col = int(x_cm / CELL_SIZE_CM)
        row = int(y_cm / CELL_SIZE_CM)
        if 0 <= row < self.dim and 0 <= col < self.dim:
            return row, col
        return None

    def _bresenham(self, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
        """Return all grid cells on the line from (r0,c0) to (r1,c1)."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc > dr:
            err = dc / 2
            while c != c1:
                cells.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr / 2
            while r != r1:
                cells.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        cells.append((r1, c1))
        return cells

    def update(
        self,
        robot_x_cm: float,
        robot_y_cm: float,
        theta_deg: float,
        sensor_dist_cm: float,
        sensor_angle_offset_deg: float = 0.0,
        sensor_forward_offset_cm: float = 0.0,
        sensor_lateral_offset_cm: float = 0.0,
    ) -> None:
        """
        Update grid from one ultrasonic reading.
        - Cells along the ray up to the hit are marked free
        - The cell at the hit is marked occupied
        - Readings outside valid range are ignored
        """
        if not (MIN_ULTRASONIC_CM <= sensor_dist_cm <= MAX_ULTRASONIC_CM):
            return

        theta_rad = math.radians(theta_deg)

        # Sensor world position
        sx = robot_x_cm + sensor_forward_offset_cm * math.cos(theta_rad) \
                        - sensor_lateral_offset_cm * math.sin(theta_rad)
        sy = robot_y_cm + sensor_forward_offset_cm * math.sin(theta_rad) \
                        + sensor_lateral_offset_cm * math.cos(theta_rad)

        # Obstacle world position
        ray_angle_rad = math.radians(theta_deg + sensor_angle_offset_deg)
        ox = sx + sensor_dist_cm * math.cos(ray_angle_rad)
        oy = sy + sensor_dist_cm * math.sin(ray_angle_rad)

        start_cell = self._cm_to_cell(sx, sy)
        end_cell   = self._cm_to_cell(ox, oy)

        if start_cell is None:
            return

        with self._lock:
            if end_cell is not None:
                ray_cells = self._bresenham(start_cell[0], start_cell[1],
                                            end_cell[0],   end_cell[1])
                # All cells except the last are free
                for cell in ray_cells[:-1]:
                    r, c = cell
                    self._grid[r, c] = max(LOG_ODDS_MIN,
                                           self._grid[r, c] + LOG_ODDS_FREE)
                # Last cell is the obstacle hit
                r, c = ray_cells[-1]
                self._grid[r, c] = min(LOG_ODDS_MAX,
                                       self._grid[r, c] + LOG_ODDS_HIT)
            else:
                # Obstacle out of bounds — just mark ray cells as free
                if start_cell:
                    ray_cells = self._bresenham(
                        start_cell[0], start_cell[1],
                        max(0, min(self.dim-1, int(oy / CELL_SIZE_CM))),
                        max(0, min(self.dim-1, int(ox / CELL_SIZE_CM))),
                    )
                    for cell in ray_cells:
                        r, c = cell
                        self._grid[r, c] = max(LOG_ODDS_MIN,
                                               self._grid[r, c] + LOG_ODDS_FREE)

    def get_probability_map(self) -> np.ndarray:
        """Return a (dim x dim) array of occupation probabilities [0..1]."""
        with self._lock:
            return 1.0 / (1.0 + np.exp(-self._grid.copy()))

    def reset(self) -> None:
        with self._lock:
            self._grid[:] = 0.0


# Global occupancy grid — shared across all robot telemetry handlers
occupancy_grid = OccupancyGrid()


# ----------------------------
# Session state
# ----------------------------
@dataclass
class ClientSession:
    client_id: int
    conn: socket.socket
    addr: tuple
    name: str = ""
    robot_id: Optional[str] = None
    state: str = "connected"
    last_heartbeat: float = field(default_factory=time.time)
    last_telemetry: Optional[dict] = None
    last_status: Optional[dict] = None
    current_path_id: Optional[str] = None
    current_waypoint_index: Optional[int] = None
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_waypoints: list[dict] = field(default_factory=list)
    pending_motion: Optional[dict] = None
    sequence_path_id: Optional[int] = None
    active_subpath_id: Optional[int] = None
    awaiting_path_ack: bool = False
    awaiting_path_complete: bool = False
    # Planned path cells for GUI overlay
    planned_path_cells: list[tuple[int, int]] = field(default_factory=list)


clients_lock = threading.Lock()
client_sessions: Dict[int, ClientSession] = {}
robots_by_id: Dict[str, int] = {}
next_client_id = 1
next_robot_path_id = 1000

# Mission state — shared, protected by clients_lock
mission_state: str = "idle"   # idle | searching | retrieving | complete
target_position: Optional[dict] = None   # {"x_cm": float, "y_cm": float}


# ----------------------------
# Networking helpers
# ----------------------------
def send_json(conn: socket.socket, message: dict) -> None:
    data = json.dumps(message) + "\n"
    conn.sendall(data.encode("utf-8"))


def recv_lines(conn: socket.socket):
    buffer = ""
    while True:
        data = conn.recv(4096)
        if not data:
            break
        buffer += data.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield line


def get_session(client_id: int) -> Optional[ClientSession]:
    with clients_lock:
        return client_sessions.get(client_id)


def get_robot_snapshot() -> dict:
    with clients_lock:
        snapshot = {}
        for robot_id, client_id in robots_by_id.items():
            session = client_sessions.get(client_id)
            if session is None:
                continue
            snapshot[robot_id] = {
                "client_id": session.client_id,
                "name": session.name,
                "state": session.state,
                "path_id": session.current_path_id,
                "waypoint_index": session.current_waypoint_index,
                "last_heartbeat": session.last_heartbeat,
                "last_telemetry": session.last_telemetry,
                "last_status": session.last_status,
                "planned_path_cells": session.planned_path_cells,
            }
        return snapshot


def print_robot_table() -> None:
    with clients_lock:
        print("\n=== Robot Table ===")
        if not client_sessions:
            print("(no clients connected)")
        for client_id, session in client_sessions.items():
            print(
                f"client_id={client_id} "
                f"robot_id={session.robot_id} "
                f"name={session.name!r} "
                f"state={session.state!r} "
                f"path_id={session.current_path_id!r} "
                f"waypoint_index={session.current_waypoint_index!r} "
                f"last_heartbeat={session.last_heartbeat:.1f}"
            )
        print("===================\n")


def send_to_robot(robot_id: str, message: dict) -> bool:
    with clients_lock:
        client_id = robots_by_id.get(robot_id)
        if client_id is None:
            print(f"[SEND] No connected session for robot_id={robot_id}")
            return False
        session = client_sessions.get(client_id)
        if session is None:
            print(f"[SEND] Session disappeared for robot_id={robot_id}")
            return False

    try:
        with session.send_lock:
            send_json(session.conn, message)
        print(f"[SEND] -> {robot_id}: {message}")
        return True
    except Exception as exc:
        print(f"[!] Failed to send to robot {robot_id}: {exc}")
        return False


def next_subpath_id() -> int:
    global next_robot_path_id
    next_robot_path_id += 1
    if next_robot_path_id > 30000:
        next_robot_path_id = 1000
    return next_robot_path_id


def clear_robot_sequence(session: ClientSession) -> None:
    session.pending_waypoints.clear()
    session.pending_motion = None
    session.sequence_path_id = None
    session.active_subpath_id = None
    session.awaiting_path_ack = False
    session.awaiting_path_complete = False
    session.planned_path_cells = []


# ----------------------------
# Grid helpers
# ----------------------------
def clamp_cell(value: int) -> int:
    return max(0, min(GRID_DIM_CELLS - 1, value))


def pose_to_cell(telemetry: dict) -> tuple[int, int] | None:
    try:
        x_cm = float(telemetry["x_cm"])
        y_cm = float(telemetry["y_cm"])
    except (KeyError, TypeError, ValueError):
        return None
    col = clamp_cell(int(x_cm // GRID_CELL_CM))
    row = clamp_cell(int(y_cm // GRID_CELL_CM))
    return row, col


def cell_center_waypoint(cell: tuple[int, int]) -> dict:
    row, col = cell
    return {
        "x_cm": col * GRID_CELL_CM + GRID_CELL_CM / 2.0,
        "y_cm": row * GRID_CELL_CM + GRID_CELL_CM / 2.0,
    }


def cm_to_cell(x_cm: float, y_cm: float) -> tuple[int, int]:
    col = clamp_cell(int(x_cm // GRID_CELL_CM))
    row = clamp_cell(int(y_cm // GRID_CELL_CM))
    return row, col


# ----------------------------
# Space-time A* planner
# ----------------------------
# Each node in the search is (row, col, timestep).
# The reservation table maps (row, col, t) -> robot_id.
# A robot can only enter a cell if no other robot has reserved it at that timestep,
# and also checks a swap-collision (two robots crossing paths in one step).
#
# This replaces the original BFS which had no time dimension and forced
# robots to move one at a time.

def heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def plan_space_time_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    reservations: dict,          # (row, col, t) -> robot_id
    robot_id: str,
    max_t: int = 200,
) -> list[tuple[int, int]] | None:
    """
    A* over (row, col, timestep) space.
    Returns a list of (row, col) cells from start to goal (inclusive),
    or None if no path found within max_t steps.
    """
    if start == goal:
        return [start]

    # heap: (f, g, row, col, t)
    start_t = 0
    heap = [(heuristic(start, goal), 0, start[0], start[1], start_t)]
    came_from = {}  # (row, col, t) -> (row, col, t) | None
    came_from[(start[0], start[1], start_t)] = None
    g_score = {(start[0], start[1], start_t): 0}

    while heap:
        f, g, row, col, t = heapq.heappop(heap)

        if (row, col) == goal:
            # Reconstruct path (spatial only)
            path = []
            state = (row, col, t)
            while state is not None:
                path.append((state[0], state[1]))
                state = came_from[state]
            path.reverse()
            return path

        if t >= max_t:
            continue

        next_t = t + 1

        # Moves: 4 directions + wait in place
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
            nr, nc = row + d_row, col + d_col

            if not (0 <= nr < GRID_DIM_CELLS and 0 <= nc < GRID_DIM_CELLS):
                continue

            next_state = (nr, nc, next_t)

            # Cell-time collision: another robot reserved this cell at next_t
            if reservations.get((nr, nc, next_t), robot_id) != robot_id:
                continue

            # Swap collision: two robots crossing each other's paths
            if (
                reservations.get((row, col, next_t), robot_id) != robot_id
                and reservations.get((nr, nc, t), robot_id) != robot_id
            ):
                continue

            new_g = g + 1
            if next_state in g_score and g_score[next_state] <= new_g:
                continue

            g_score[next_state] = new_g
            came_from[next_state] = (row, col, t)
            f_score = new_g + heuristic((nr, nc), goal)
            heapq.heappush(heap, (f_score, new_g, nr, nc, next_t))

    return None


def reserve_path(
    path: list[tuple[int, int]],
    robot_id: str,
    reservations: dict,
) -> None:
    """Write all (row, col, t) reservations for a planned path."""
    for t, cell in enumerate(path):
        reservations[(cell[0], cell[1], t)] = robot_id
    # Hold the goal cell for extra timesteps so trailing robots don't cut through
    if path:
        goal = path[-1]
        last_t = len(path) - 1
        for extra in range(1, 20):
            reservations[(goal[0], goal[1], last_t + extra)] = robot_id


def plan_paths_simultaneous(
    starts: dict,   # robot_id -> (row, col)
    goals: dict,    # robot_id -> (row, col)
) -> dict | None:
    """
    Plan collision-free paths for all robots simultaneously using
    space-time A* with a shared reservation table.

    Returns dict of robot_id -> list[(row, col)], or None if any robot
    cannot find a path.
    """
    reservations = {}
    paths = {}

    # Plan robots in order of path length (shorter paths first reduces conflicts)
    robot_ids = list(starts.keys())
    robot_ids.sort(key=lambda r: heuristic(starts[r], goals[r]))

    for robot_id in robot_ids:
        path = plan_space_time_path(
            start=starts[robot_id],
            goal=goals[robot_id],
            reservations=reservations,
            robot_id=robot_id,
        )
        if path is None:
            print(f"[PLANNER] No space-time path found for {robot_id}")
            return None

        reserve_path(path, robot_id, reservations)
        paths[robot_id] = path
        print(f"[PLANNER] {robot_id}: {starts[robot_id]} -> {goals[robot_id]}, {len(path)-1} steps")

    return paths


# ----------------------------
# Waypoint dispatch
# ----------------------------
def dispatch_next_waypoint(robot_id: str) -> bool:
    """
    Pop the next pending waypoint for this robot and send it.
    No longer checks whether other robots are busy — simultaneous movement
    is now handled by the space-time planner at planning time.
    """
    with clients_lock:
        client_id = robots_by_id.get(robot_id)
        if client_id is None:
            return False

        session = client_sessions.get(client_id)
        if session is None:
            return False

        if session.awaiting_path_ack or session.awaiting_path_complete:
            return False

        if not session.pending_waypoints:
            session.pending_motion = None
            session.sequence_path_id = None
            session.active_subpath_id = None
            return False

        waypoint = session.pending_waypoints.pop(0)
        subpath_id = next_subpath_id()
        message = {
            "type": "path_assignment",
            "robot_id": robot_id,
            "path_id": subpath_id,
            "replace_existing": True,
            "waypoints": [waypoint],
        }
        if session.pending_motion is not None:
            message["motion"] = dict(session.pending_motion)

        session.active_subpath_id = subpath_id
        session.current_path_id = str(subpath_id)
        session.current_waypoint_index = 0
        session.awaiting_path_ack = True
        session.awaiting_path_complete = True

    ok = send_to_robot(robot_id, message)
    if not ok:
        with clients_lock:
            client_id = robots_by_id.get(robot_id)
            session = client_sessions.get(client_id) if client_id is not None else None
            if session is not None:
                session.pending_waypoints.insert(0, waypoint)
                session.active_subpath_id = None
                session.awaiting_path_ack = False
                session.awaiting_path_complete = False
        return False

    print(f"[SEQUENCE] dispatched subpath {subpath_id} to {robot_id} -> {waypoint}")
    return True


def maybe_dispatch_waiting_sequences() -> None:
    """
    Dispatch the next waypoint for EVERY robot that has pending work
    and is not currently executing. Robots now move in parallel.
    """
    with clients_lock:
        robot_ids = [
            session.robot_id
            for session in client_sessions.values()
            if session.robot_id
            and session.pending_waypoints
            and not session.awaiting_path_complete
            and not session.awaiting_path_ack
        ]

    for robot_id in robot_ids:
        dispatch_next_waypoint(robot_id)


def queue_robot_path(message: dict) -> bool:
    robot_id = message.get("robot_id")
    if not robot_id:
        return False

    waypoints = []
    for waypoint in message.get("waypoints", []):
        x_cm = waypoint.get("x_cm")
        y_cm = waypoint.get("y_cm")
        if x_cm is None or y_cm is None:
            continue
        waypoints.append({"x_cm": float(x_cm), "y_cm": float(y_cm)})

    if not waypoints:
        print(f"[SEQUENCE] Refusing to queue empty path for {robot_id}")
        return False

    with clients_lock:
        client_id = robots_by_id.get(robot_id)
        if client_id is None:
            return False
        session = client_sessions.get(client_id)
        if session is None:
            return False

        session.pending_waypoints = waypoints
        session.pending_motion = dict(message["motion"]) if isinstance(message.get("motion"), dict) else None
        session.sequence_path_id = int(message.get("path_id", next_subpath_id()))
        session.active_subpath_id = None
        session.awaiting_path_ack = False
        session.awaiting_path_complete = False

    maybe_dispatch_waiting_sequences()
    return True


# ----------------------------
# Coordinated traverse (upgraded)
# ----------------------------
def start_coordinated_traverse(message: dict) -> bool:
    """
    Two-robot coordinated traverse using space-time A*.
    Both robots plan and move simultaneously — no turn-taking.
    """
    robots = message.get("robots", [])
    if len(robots) != 2:
        print("[COORD] Expected exactly two robots")
        return False

    robot_one_id = str(robots[0].get("robot_id", "")).strip()
    robot_two_id = str(robots[1].get("robot_id", "")).strip()

    if not robot_one_id or not robot_two_id or robot_one_id == robot_two_id:
        print("[COORD] Need two distinct robot IDs")
        return False

    with clients_lock:
        s1 = client_sessions.get(robots_by_id.get(robot_one_id))
        s2 = client_sessions.get(robots_by_id.get(robot_two_id))

        if s1 is None or s2 is None:
            print("[COORD] One or both robots not connected")
            return False

        start_one = pose_to_cell(s1.last_telemetry or {})
        start_two = pose_to_cell(s2.last_telemetry or {})
        if start_one is None or start_two is None:
            print("[COORD] Need telemetry from both robots before planning")
            return False

        goal_one = (clamp_cell(int(robots[0]["goal_row"])), clamp_cell(int(robots[0]["goal_col"])))
        goal_two = (clamp_cell(int(robots[1]["goal_row"])), clamp_cell(int(robots[1]["goal_col"])))

        if start_one == start_two:
            print("[COORD] Robots in same cell — separate them first")
            return False
        if goal_one == goal_two:
            print("[COORD] Robots cannot share a goal cell")
            return False

        clear_robot_sequence(s1)
        clear_robot_sequence(s2)

    paths = plan_paths_simultaneous(
        starts={robot_one_id: start_one, robot_two_id: start_two},
        goals={robot_one_id: goal_one, robot_two_id: goal_two},
    )

    if paths is None:
        print("[COORD] Space-time planner failed to find collision-free paths")
        return False

    with clients_lock:
        s1 = client_sessions[robots_by_id[robot_one_id]]
        s2 = client_sessions[robots_by_id[robot_two_id]]

        s1.pending_waypoints = [cell_center_waypoint(c) for c in paths[robot_one_id][1:]]
        s1.planned_path_cells = paths[robot_one_id]
        s1.sequence_path_id = next_subpath_id()
        s1.active_subpath_id = None
        s1.awaiting_path_ack = False
        s1.awaiting_path_complete = False

        s2.pending_waypoints = [cell_center_waypoint(c) for c in paths[robot_two_id][1:]]
        s2.planned_path_cells = paths[robot_two_id]
        s2.sequence_path_id = next_subpath_id()
        s2.active_subpath_id = None
        s2.awaiting_path_ack = False
        s2.awaiting_path_complete = False

    # Dispatch both robots simultaneously
    maybe_dispatch_waiting_sequences()
    return True


# ----------------------------
# Retrieval mission
# ----------------------------
def start_retrieval_mission(x_cm: float, y_cm: float) -> bool:
    """
    Robot A navigates to the target (the car door).
    Robot B moves to a staging position out of the way.
    Both move simultaneously via space-time planning.
    Called by vision pipeline or GUI button.
    """
    global mission_state, target_position

    with clients_lock:
        robot_ids = list(robots_by_id.keys())

    if len(robot_ids) < 1:
        print("[MISSION] No robots connected")
        return False

    robot_a_id = "robot_A"
    robot_b_id = "robot_B"

    with clients_lock:
        s_a = client_sessions.get(robots_by_id.get(robot_a_id))
        s_b = client_sessions.get(robots_by_id.get(robot_b_id))

    target_cell = cm_to_cell(x_cm, y_cm)
    staging_cell = cm_to_cell(STAGING_X_CM, STAGING_Y_CM)

    starts = {}
    goals = {}

    if s_a and s_a.last_telemetry:
        start_a = pose_to_cell(s_a.last_telemetry)
        if start_a:
            starts[robot_a_id] = start_a
            goals[robot_a_id] = target_cell

    if s_b and s_b.last_telemetry:
        start_b = pose_to_cell(s_b.last_telemetry)
        if start_b and start_b != staging_cell:
            starts[robot_b_id] = start_b
            goals[robot_b_id] = staging_cell

    if not starts:
        print("[MISSION] No robot telemetry available to plan from")
        return False

    paths = plan_paths_simultaneous(starts=starts, goals=goals)

    if paths is None:
        print("[MISSION] Planner could not find collision-free paths for mission")
        return False

    with clients_lock:
        mission_state = "retrieving"
        target_position = {"x_cm": x_cm, "y_cm": y_cm}

        for rid, path in paths.items():
            cid = robots_by_id.get(rid)
            session = client_sessions.get(cid) if cid else None
            if session is None:
                continue
            clear_robot_sequence(session)
            session.pending_waypoints = [cell_center_waypoint(c) for c in path[1:]]
            session.planned_path_cells = path
            session.sequence_path_id = next_subpath_id()
            session.active_subpath_id = None
            session.awaiting_path_ack = False
            session.awaiting_path_complete = False

    print(f"[MISSION] Retrieval mission started — target ({x_cm:.0f}, {y_cm:.0f}) cm")
    gui.update_mission_state("retrieving", x_cm, y_cm)
    maybe_dispatch_waiting_sequences()
    return True


# ----------------------------
# GUI command callback
# ----------------------------
def gui_command_sender(message_obj) -> None:
    try:
        message = message_obj if isinstance(message_obj, dict) else message_obj.to_dict()
        robot_id = message.get("robot_id")

        if message.get("type") == "coordinated_traverse":
            ok = start_coordinated_traverse(message)
            print("[GUI SEND] Coordinated traverse", "started" if ok else "FAILED")
            return

        if message.get("type") == "retrieval_mission":
            ok = start_retrieval_mission(
                float(message.get("x_cm", 200)),
                float(message.get("y_cm", 200)),
            )
            print("[GUI SEND] Retrieval mission", "started" if ok else "FAILED")
            return

        if not robot_id:
            print("[GUI SEND] Refusing to send message with no robot_id")
            return

        if message.get("type") == "path_assignment":
            ok = queue_robot_path(message)
        else:
            if message.get("type") == "stop":
                with clients_lock:
                    cid = robots_by_id.get(robot_id)
                    session = client_sessions.get(cid) if cid else None
                    if session is not None:
                        clear_robot_sequence(session)
            ok = send_to_robot(robot_id, message)

        print(f"[GUI SEND] {'Sent' if ok else 'FAILED'} {message.get('type')} to {robot_id}")

    except Exception as exc:
        print(f"[GUI SEND] Error: {exc}")


# Initialize GUI
gui = TelemetryGUI(command_sender=gui_command_sender)


# ----------------------------
# Session / identity helpers
# ----------------------------
def touch_session(client_id: int) -> None:
    with clients_lock:
        session = client_sessions.get(client_id)
        if session:
            session.last_heartbeat = time.time()


def bind_identity_from_message(client_id: int, msg: dict) -> None:
    with clients_lock:
        session = client_sessions[client_id]

        if "name" in msg:
            session.name = str(msg["name"])
        elif "client_name" in msg:
            session.name = str(msg["client_name"])

        if "robot_id" in msg and msg["robot_id"] is not None:
            robot_id = str(msg["robot_id"])
            session.robot_id = robot_id
            robots_by_id[robot_id] = client_id

        if "state" in msg and msg["state"] is not None:
            session.state = str(msg["state"])

        if "path_id" in msg and msg["path_id"] is not None:
            session.current_path_id = str(msg["path_id"])

        if "waypoint_index" in msg and msg["waypoint_index"] is not None:
            try:
                session.current_waypoint_index = int(msg["waypoint_index"])
            except (TypeError, ValueError):
                pass


# ----------------------------
# Message handlers
# ----------------------------
def handle_hello(client_id: int, msg: dict, conn: socket.socket) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)
    session = get_session(client_id)
    print(f"[Client {client_id}] registered name={session.name!r}, robot_id={session.robot_id!r}")
    send_json(conn, {"type": "ack", "for": "hello", "client_id": client_id, "robot_id": session.robot_id})
    print_robot_table()


def handle_telemetry(client_id: int, msg: dict) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)
    with clients_lock:
        session = client_sessions[client_id]
        session.last_telemetry = msg
    if session.robot_id:
        gui.update_robot(session.robot_id, msg)

    # Update occupancy grid from ultrasonic readings
    x_cm    = msg.get("x_cm")
    y_cm    = msg.get("y_cm")
    theta   = msg.get("theta_deg")
    front   = msg.get("front_ultrasonic_cm")
    left    = msg.get("left_ultrasonic_cm")

    if x_cm is not None and y_cm is not None and theta is not None:
        if front is not None:
            occupancy_grid.update(
                float(x_cm), float(y_cm), float(theta),
                float(front),
                sensor_angle_offset_deg=0.0,
                sensor_forward_offset_cm=9.5,
                sensor_lateral_offset_cm=0.0,
            )
        if left is not None:
            occupancy_grid.update(
                float(x_cm), float(y_cm), float(theta),
                float(left),
                sensor_angle_offset_deg=90.0,
                sensor_forward_offset_cm=0.0,
                sensor_lateral_offset_cm=16.0,
            )
        # Push updated map to GUI
        gui.update_occupancy_map(occupancy_grid.get_probability_map())

    robot_label = session.robot_id or f"client_{client_id}"
    print(f"[{robot_label}] telemetry: {msg}")


def handle_status(client_id: int, msg: dict) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)
    with clients_lock:
        session = client_sessions[client_id]
        session.last_status = msg
        merged = dict(session.last_telemetry or {})
        merged.update(msg)
    if session.robot_id:
        gui.update_robot(session.robot_id, merged)
    print(f"[{session.robot_id or f'client_{client_id}'}] status: {msg}")


def handle_path_event(client_id: int, msg: dict) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)

    with clients_lock:
        session = client_sessions[client_id]
        merged = dict(session.last_telemetry or {})
        merged.update(msg)

        event_type = str(msg.get("type", ""))
        if event_type == "path_started":
            merged["state"] = "executing_path"
            session.state = "executing_path"
        elif event_type == "waypoint_reached":
            merged["state"] = "waypoint_reached"
            session.state = "waypoint_reached"
        elif event_type == "path_complete":
            merged["state"] = "idle"
            session.state = "idle"
            session.current_waypoint_index = None
            session.current_path_id = None
            session.awaiting_path_complete = False
            session.active_subpath_id = None

        session.last_status = merged

    if session.robot_id:
        gui.update_robot(session.robot_id, merged)

    print(f"[{session.robot_id or f'client_{client_id}'}] {msg.get('type')}: {msg}")

    if msg.get("type") == "path_complete":
        if session.robot_id:
            dispatch_next_waypoint(session.robot_id)
        maybe_dispatch_waiting_sequences()


def handle_ack(client_id: int, msg: dict) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)
    with clients_lock:
        session = client_sessions.get(client_id)
        if session is not None and msg.get("for") == "path_assignment":
            ack_path_id = msg.get("path_id")
            if ack_path_id is None or session.active_subpath_id is None:
                session.awaiting_path_ack = False
            else:
                try:
                    if int(ack_path_id) == session.active_subpath_id:
                        session.awaiting_path_ack = False
                except (TypeError, ValueError):
                    session.awaiting_path_ack = False
    print(f"[{session.robot_id if session and session.robot_id else f'client_{client_id}'}] ack: {msg}")


def handle_heartbeat(client_id: int, conn: socket.socket, msg: dict) -> None:
    touch_session(client_id)
    bind_identity_from_message(client_id, msg)
    session = get_session(client_id)
    send_json(conn, {
        "type": "heartbeat_ack",
        "robot_id": session.robot_id if session else None,
        "server_t": time.time(),
    })


def handle_target_located(msg: dict) -> None:
    """
    Called when vision.py sends a target_located message.
    Automatically kicks off the retrieval mission.
    """
    global mission_state, target_position

    x_cm = msg.get("x_cm")
    y_cm = msg.get("y_cm")

    if x_cm is None or y_cm is None:
        print("[VISION] target_located missing coordinates")
        return

    print(f"[VISION] Target located at ({x_cm:.1f}, {y_cm:.1f}) cm — starting retrieval mission")

    with clients_lock:
        if mission_state == "retrieving":
            print("[VISION] Mission already in progress — ignoring duplicate target_located")
            return
        mission_state = "searching"

    gui.update_mission_state("target_found", float(x_cm), float(y_cm))
    start_retrieval_mission(float(x_cm), float(y_cm))


# ----------------------------
# Client thread
# ----------------------------
def handle_client(client_id: int, conn: socket.socket, addr) -> None:
    print(f"[+] Client {client_id} connected from {addr}")

    try:
        send_json(conn, {"type": "hello_ack", "client_id": client_id, "message": "connected"})

        for line in recv_lines(conn):
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Client {client_id}] Bad JSON: {line}")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "hello":
                handle_hello(client_id, msg, conn)
            elif msg_type == "telemetry":
                handle_telemetry(client_id, msg)
            elif msg_type == "heartbeat":
                handle_heartbeat(client_id, conn, msg)
            elif msg_type == "status":
                handle_status(client_id, msg)
            elif msg_type in {"path_started", "waypoint_reached", "path_complete"}:
                handle_path_event(client_id, msg)
            elif msg_type == "ack":
                handle_ack(client_id, msg)
            elif msg_type == "target_located":
                handle_target_located(msg)
            else:
                print(f"[Client {client_id}] unknown message type: {msg_type}")
                send_json(conn, {"type": "error", "message": f"unknown message type: {msg_type}"})

    except ConnectionResetError:
        print(f"[!] Client {client_id} connection reset")
    except Exception as exc:
        print(f"[!] Client {client_id} error: {exc}")
    finally:
        with clients_lock:
            session = client_sessions.pop(client_id, None)
            if session and session.robot_id:
                if robots_by_id.get(session.robot_id) == client_id:
                    robots_by_id.pop(session.robot_id, None)
        conn.close()
        print(f"[-] Client {client_id} disconnected")
        print_robot_table()


# ----------------------------
# Accept loop
# ----------------------------
def accept_loop(server_sock: socket.socket) -> None:
    global next_client_id
    while True:
        conn, addr = server_sock.accept()
        with clients_lock:
            if len(client_sessions) >= MAX_CLIENTS:
                send_json(conn, {"type": "error", "message": "server full"})
                conn.close()
                continue
            client_id = next_client_id
            next_client_id += 1
            client_sessions[client_id] = ClientSession(client_id=client_id, conn=conn, addr=addr)
        threading.Thread(target=handle_client, args=(client_id, conn, addr), daemon=True).start()
        print_robot_table()


# ----------------------------
# Server main
# ----------------------------
def server_main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        print(f"Server listening on {HOST}:{PORT}")
        accept_loop(server_sock)


def main() -> None:
    server_thread = threading.Thread(target=server_main, daemon=True)
    server_thread.start()
    gui.run()


if __name__ == "__main__":
    main()