Detect all graspable objects on the table surface.
Also detect the robot hand/gripper when visible.
Ignore the background, the table itself, and non-object clutter.

Handle rules:
- Each handle must be unique and stable: color_type_NN (e.g. red_cube_01, green_bottle_01, yellow_ball_02).
- Use robot hand handle format: robot_hand_NN (e.g. robot_hand_01).

Graspability rules:
- Pickable objects must have "is_graspable": true.
- Robot hand/gripper must have "is_graspable": false.

Output rules:
- JSON only (no prose, no markdown fences).
- If the robot hand is visible, include it in "detections".
- If the robot hand is not visible, do not hallucinate it.

Return this exact JSON shape:
{"scene":"...","detections":[{"handle":"...","label":"...","bounding_box":[x1,y1,x2,y2],"is_graspable":true}]}
