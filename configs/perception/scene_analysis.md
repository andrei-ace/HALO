Detect all graspable objects on the table surface. Ignore the background, the table itself, and any non-object clutter.
Each handle must be unique: color_type_NN (e.g. red_cube_01, green_bottle_01, yellow_ball_02).
JSON only:
{"scene":"...","detections":[{"handle":"...","label":"...","bounding_box":[x1,y1,x2,y2]}]}
