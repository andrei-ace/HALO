Detect all objects on the table: graspable items (cubes, balls, bottles, etc.) AND containers (trays, bowls, bins).
Also detect the robot hand/gripper when visible. Ignore the table itself and background.

Handle format: {color}_{type}_{nn} using the object's actual visible color, e.g. green_cube_01, yellow_tray_01.
Label: short human description (e.g. "small green cube"), NOT a copy of the handle.
Scene: one sentence describing the layout.
Do not invent objects not in the image.
