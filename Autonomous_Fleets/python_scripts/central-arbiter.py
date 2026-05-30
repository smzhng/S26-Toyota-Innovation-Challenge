import json
from collections import deque
import heapq
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from gui import TelemetryGUI

HOST = "0.0.0.0"
PORT = 9000
MAX_CLIENTS = 10
GRID_CELL_CM = 10.0
GRID_DIM_CELLS = 40


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

# Maze walls — set of (row, col) grid cells the planner cannot enter
# Updated live from the GUI; no lock needed (GIL protects simple set replacement)
wall_cells: set[tuple[int, int]] = set()


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

            # Block cardboard wall cells
            if (nr, nc) in wall_cells:
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


# ----------------------------
# Waypoint dispatch
# ----------------------------
def dispatch_next_waypoint(robot_id: str) -> bool:
    """Pop the next pending waypoint for this robot and send it."""
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
    """Dispatch the next waypoint for any robot that has pending work and is idle."""
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
# Pipe mission
# ----------------------------
def start_retrieval_mission(x_cm: float, y_cm: float) -> bool:
    """
    Plan a wall-avoiding A* path and send robot_A to the black pipe.
    Called by vision pipeline or the GUI "Send to Pipe" button.
    """
    global mission_state, target_position

    robot_a_id = "robot_A"

    with clients_lock:
        s_a = client_sessions.get(robots_by_id.get(robot_a_id))

    if s_a is None or not s_a.last_telemetry:
        print("[MISSION] robot_A not connected or no telemetry yet")
        return False

    start_cell = pose_to_cell(s_a.last_telemetry)
    if start_cell is None:
        print("[MISSION] Cannot determine robot_A position from telemetry")
        return False

    target_cell = cm_to_cell(x_cm, y_cm)
    print(f"[MISSION] Planning: robot at cell {start_cell} "
          f"({s_a.last_telemetry.get('x_cm','?'):.1f}, {s_a.last_telemetry.get('y_cm','?'):.1f}) cm "
          f"-> pipe cell {target_cell} ({x_cm:.0f}, {y_cm:.0f}) cm  walls={len(wall_cells)}")

    path = plan_space_time_path(
        start=start_cell,
        goal=target_cell,
        reservations={},
        robot_id=robot_a_id,
    )

    if path is None:
        print("[MISSION] A* could not find a path — check that walls don't block all routes")
        return False

    waypoints = [cell_center_waypoint(c) for c in path[1:]]
    print(f"[MISSION] Path found: {len(path)-1} steps")
    for i, wp in enumerate(waypoints):
        print(f"  waypoint {i+1}: ({wp['x_cm']:.0f}, {wp['y_cm']:.0f}) cm")
    if not waypoints:
        print("[MISSION] WARNING: path is 0 steps — robot is already in the target cell")

    with clients_lock:
        mission_state = "retrieving"
        target_position = {"x_cm": x_cm, "y_cm": y_cm}
        cid = robots_by_id.get(robot_a_id)
        session = client_sessions.get(cid) if cid else None
        if session is None:
            return False
        clear_robot_sequence(session)
        session.pending_waypoints = waypoints
        session.planned_path_cells = path
        session.sequence_path_id = next_subpath_id()
        session.active_subpath_id = None
        session.awaiting_path_ack = False
        session.awaiting_path_complete = False

    print(f"[MISSION] Dispatching first waypoint to robot_A")
    gui.update_mission_state("retrieving", x_cm, y_cm)
    dispatch_next_waypoint(robot_a_id)
    return True


# ----------------------------
# GUI command callback
# ----------------------------
def gui_command_sender(message_obj) -> None:
    try:
        message = message_obj if isinstance(message_obj, dict) else message_obj.to_dict()
        robot_id = message.get("robot_id")

        if message.get("type") == "set_walls":
            global wall_cells
            wall_cells = {tuple(c) for c in message.get("wall_cells", [])}
            print(f"[GUI] Maze walls updated: {len(wall_cells)} blocked cells")
            return

        if message.get("type") == "retrieval_mission":
            ok = start_retrieval_mission(
                float(message.get("x_cm", 200)),
                float(message.get("y_cm", 200)),
            )
            print("[GUI SEND] Pipe mission", "started" if ok else "FAILED")
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
    """Called when vision.py confirms the black pipe location."""
    global mission_state

    x_cm = msg.get("x_cm")
    y_cm = msg.get("y_cm")

    if x_cm is None or y_cm is None:
        print("[VISION] target_located missing coordinates")
        return

    print(f"[VISION] Pipe located at ({x_cm:.1f}, {y_cm:.1f}) cm — planning maze route")

    with clients_lock:
        if mission_state == "retrieving":
            print("[VISION] Mission already in progress — ignoring duplicate")
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
