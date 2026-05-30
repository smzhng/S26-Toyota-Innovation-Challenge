# Milestone 3 SOP — Maze Navigation & Pipe Retrieval

## Overview

Single robot (`robot_A`) navigates a cardboard-wall maze, locates a black pipe via computer vision, plans an obstacle-avoiding path, and picks it up with the gripper.

## Prerequisites
- PRIZM controller connected to laptop via USB
- Arduino IDE installed with PRIZM library
- Python 3 with dependencies installed (`pyserial`, `python-dotenv`, `matplotlib`, `opencv-python`, `numpy`)

---

## Step 1 — Find the Serial Port

Plug in the PRIZM, then run:

```bash
ls /dev/cu.*
```

Look for a name like `/dev/cu.usbserial-XXXXXXXX`. Open `robot_a.env` and set:

```
LOCAL_SERIAL_PORT="/dev/cu.usbserial-XXXXXXXX"
```

---

## Step 2 — Upload Firmware

1. Open Arduino IDE
2. Open `prizm_firmware/telemetry_and_communicate_to_aribiter/telemetry_and_communicate_to_aribiter.ino`
3. Set **Tools → Board → Arduino Uno**
4. Set **Tools → Port** to match the port from Step 1
5. Click **Upload**
6. After upload completes, press the **green START button** on the PRIZM

---

## Step 3 — Start the Central Arbiter

Open a terminal:

```bash
cd python_scripts
python3 central-arbiter.py
```

The GUI dashboard opens and listens on port 9000. Leave this running.

---

## Step 4 — Set the Robot's Starting Pose

**This is the most common reason the robot doesn't move.** The firmware tracks position via wheel odometry starting from a hardcoded origin. You must tell the software where the robot is physically placed.

1. In the GUI under **"Robot Starting Pose"**, enter the robot's X and Y (in cm from the arena corner) and its heading θ° (0° = facing right, 90° = facing up)
2. Select **robot_A** in the "Target Robot" dropdown
3. Click **"Set Robot Start Pose"**

The robot's dot on the arena plot should jump to that position immediately (next telemetry tick).

> **Default values**: X=200, Y=100, θ=90. If you always place the robot at the same spot you can leave these and skip this step.

> **Re-uploading the firmware?** You can also change the defaults directly in the `.ino`:
> ```cpp
> float x_cm = 200.0;   // ← change this
> float y_cm = 100.0;   // ← change this
> float theta_rad = PI / 2.0;  // PI/2 = 90°
> ```

---

## Step 6 — Draw the Maze Walls

After setting the start pose, mark the cardboard walls on the arena grid:

1. In the GUI, click **"Edit Walls: OFF"** — it toggles to **"Edit Walls: ON"** and the plot title turns red
2. Click each 10 cm grid cell on the arena plot that corresponds to a cardboard wall
   - Clicked cells turn dark gray
   - Click again to unmark a cell
3. Click **"Edit Walls: OFF"** again to exit edit mode

The A* planner will route around every marked cell. If you set walls wrong, click **"Clear All Walls"** and re-draw.

---

## Step 7 — Start the Robot Client

Open a second terminal:

```bash
cd python_scripts
python3 client.py --env ../robot_a.env
```

Expected output:
```
[COMMUNICATION] Using SERIAL /dev/cu.usbmodemXXXXX
Connected to server 127.0.0.1:9000
[CLIENT->TCP sent] hello for robot_A
[SERIAL->TCP sent] telemetry
...
```

`robot_A` will appear in the GUI table and on the arena plot.

---

## Step 8 — Locate the Pipe

### Option A — Automatic (camera)

Open a third terminal:

```bash
cd python_scripts
python3 vision.py --env ../robot_a.env
```

Two windows open: **Detection** (live feed) and **Mask** (white = detected).

Tune the HSV sliders until only the black pipe shows as white in the mask. Press **`s`** to save settings for next time.

Once the pipe is visible for 8 consecutive frames, it sends the coordinates to the arbiter automatically and the robot starts navigating.

| Key | Action |
|---|---|
| `s` | Save current HSV values to `hsv_config.json` |
| `l` | Load saved HSV values |
| `t` | Manually trigger with current detection |
| `r` | Reset so it can trigger again |
| `q` | Quit |

### Option B — Manual

In the GUI under **"Pipe Mission"**, enter the pipe's X and Y coordinates (in cm from the arena origin) and click **"Send Robot to Pipe"**.

---

## Step 9 — Watch the Run

- The arena plot shows the planned path as a dashed line and the robot's live trail
- The robot state cycles: `idle` → `executing_path` → `waypoint_reached` → ... → `idle`
- Mission status banner updates: **Navigating to pipe** while moving, **Pipe reached** when done

---

## Step 10 — Pick Up the Pipe

When the robot reaches the pipe and returns to `idle`:

1. Select **robot_A** in the "Target Robot" dropdown
2. Click **"Toggle Gripper"** to close the gripper

To abort at any point, click **"Stop"**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | Start the arbiter (Step 3) before the client |
| Serial error on client start | Verify port in `robot_a.env` matches `ls /dev/cu.*`; confirm PRIZM START button is pressed |
| Robot in GUI but ignores commands | Confirm `ROBOT_ID` in the `.ino` matches `robot_A` in `robot_a.env` |
| "Navigating to pipe" shows but robot doesn't move | **Step 1**: Check arbiter terminal — look for `[MISSION] Path found: X steps`. If X=0, the robot is already in the pipe's grid cell (re-check coordinates). If X>0, look for `[SEQUENCE] dispatched subpath` — if missing, it's a threading issue, restart arbiter. If dispatched, the problem is hardware — try "Send Straight Test" to verify motors work. |
| "Send Straight Test" doesn't move the robot | Motors aren't responding — check PRIZM power, green START button pressed, correct serial port in `robot_a.env` |
| Robot moves but goes the wrong direction/distance | Wheel calibration needed — adjust `WHEEL_DIAMETER_CM` and `WHEEL_BASE_CM` in the `.ino` and re-upload |
| A* returns "no path found" | A wall cell is blocking all routes — clear walls and re-draw, leaving a passage through the maze |
| Pipe not detected by vision | Lower V max and S max sliders; black needs V 0–60, S 0–80. Press `s` to save once it looks right |
| Robot overshoots the pipe | Tune `WHEEL_DIAMETER_CM` and `WHEEL_BASE_CM` in the `.ino` to match actual robot dimensions |
| Telemetry reaches client but not GUI | Check TCP logs in both terminals for JSON parse errors |
