"""
build_mjcf.py
--------------
Generates the MuJoCo MJCF XML for the maze: static wall boxes + pillar
cylinders (from maze_layout.py) plus a single free-sliding point-mass robot
(two orthogonal slide joints, no orientation DOF -- matches the paper's
single-integrator model dot p = u exactly).

The robot is driven KINEMATICALLY: each control step we write the safety-
filtered position update directly into qpos and call mj_forward (not
mj_step) to refresh contacts/collision geometry. MuJoCo here is used for
what it is uniquely good at and what the user asked for -- an independent,
physically grounded collision/contact ground truth and 3-D-quality
rendering of the maze -- while the *control law* stays exactly the
relative-degree-one single-integrator CBF-QP the manuscript derives
(H = I, a_i = 2(p-o_i), b_i = -alpha h_i(p)), so every identity and
closed-form result validated in the pilot study applies unchanged.
"""
import maze_layout as ml

ROBOT_Z = 0.25
WALL_HEIGHT = 0.5


def build_mjcf_xml():
    parts = []
    parts.append('<mujoco model="safe_maze">')
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

    # Body origin at (0,0,z): the two slide joints (jx, jy) are relative to
    # this origin, so qpos == absolute world (x, y) directly, matching how
    # the Python env treats self.pos everywhere (CBF filter, reward, obs).
    # (An earlier version placed the body at ml.START and wrote absolute
    # positions into qpos anyway, silently offsetting every MuJoCo
    # collision check by -START -- caught via a spurious contact while
    # debugging a training plateau, fixed here.)
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
    with open("maze.xml", "w") as f:
        f.write(xml)
    print("wrote maze.xml, %d chars" % len(xml))
