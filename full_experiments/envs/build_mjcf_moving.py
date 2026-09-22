"""
build_mjcf_moving.py
---------------------
Same maze geometry as build_mjcf.py (main maze, maze_layout.py), plus one
extra MuJoCo *mocap* body representing the moving "human" obstacle. A mocap
body has no physics/joints of its own -- each step the environment writes
its prescribed, fully-known position directly into data.mocap_pos -- so
MuJoCo still gives an independent, physically grounded contact/collision
ground truth against the human, exactly as it does against the static
walls and pillars, while the human's trajectory itself stays a simple,
known closed-form function of time (see human_obstacle.py) rather than
anything MuJoCo simulates.
"""
import maze_layout as ml
from human_obstacle import HUMAN_RADIUS, human_state

ROBOT_Z = 0.25
WALL_HEIGHT = 0.5


def build_mjcf_xml():
    parts = []
    parts.append('<mujoco model="safe_maze_moving">')
    parts.append('  <option gravity="0 0 -9.81" timestep="0.01"/>')
    parts.append('  <visual><global offwidth="900" offheight="900"/></visual>')
    parts.append('  <worldbody>')
    parts.append('    <light diffuse=".8 .8 .8" pos="0 0 8" dir="0 0 -1"/>')
    parts.append('    <geom name="floor" type="plane" size="10 10 0.1" '
                  'rgba="0.95 0.95 0.93 1" contype="2" conaffinity="2"/>')

    for k, (cx, cy, hx, hy) in enumerate(ml.wall_boxes_merged()):
        parts.append(
            f'    <geom name="wall_{k}" type="box" pos="{cx:.4f} {cy:.4f} {WALL_HEIGHT/2:.3f}" '
            f'size="{hx:.4f} {hy:.4f} {WALL_HEIGHT/2:.3f}" rgba="0.32 0.32 0.30 1" '
            f'contype="1" conaffinity="1"/>')

    for k, (cx, cy, r) in enumerate(ml.PILLARS):
        parts.append(
            f'    <geom name="pillar_{k}" type="cylinder" pos="{cx:.4f} {cy:.4f} {WALL_HEIGHT/2:.3f}" '
            f'size="{r:.4f} {WALL_HEIGHT/2:.3f}" rgba="0.89 0.29 0.28 1" '
            f'contype="1" conaffinity="1"/>')

    gx, gy = ml.GOAL
    parts.append(
        f'    <geom name="goal_marker" type="cylinder" pos="{gx:.3f} {gy:.3f} 0.02" '
        f'size="0.28 0.02" rgba="0.93 0.63 0.0 0.55" contype="0" conaffinity="0"/>')

    # Moving human: a mocap body (no joints), initial position at t=0.
    hx0, hy0 = human_state(0.0)[0]
    parts.append(f'    <body name="human" mocap="true" pos="{hx0:.4f} {hy0:.4f} {WALL_HEIGHT/2:.3f}">')
    parts.append(
        f'      <geom name="human_geom" type="cylinder" size="{HUMAN_RADIUS:.4f} {WALL_HEIGHT/2:.3f}" '
        f'rgba="0.55 0.20 0.65 0.85" contype="1" conaffinity="1"/>')
    parts.append('    </body>')

    parts.append('    <body name="robot" pos="0 0 %.3f">' % ROBOT_Z)
    parts.append('      <joint name="jx" type="slide" axis="1 0 0" limited="false" damping="0"/>')
    parts.append('      <joint name="jy" type="slide" axis="0 1 0" limited="false" damping="0"/>')
    parts.append(
        f'      <geom name="robot_geom" type="sphere" size="{ml.ROBOT_RADIUS:.3f}" '
        f'rgba="0.16 0.47 0.84 1" contype="1" conaffinity="1" mass="1.0"/>')
    parts.append('    </body>')
    parts.append('  </worldbody>')
    parts.append('</mujoco>')
    return "\n".join(parts)


if __name__ == "__main__":
    xml = build_mjcf_xml()
    with open("maze_moving.xml", "w") as f:
        f.write(xml)
    print("wrote maze_moving.xml, %d chars" % len(xml))
