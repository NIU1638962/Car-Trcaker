import numpy as np

def track_vehicles(centroids, previous_centroids, max_distance=50):
    matches = []
    for c in centroids:
        distances = [np.linalg.norm(np.array(c) - np.array(pc)) for pc in previous_centroids]
        if min(distances) < max_distance:
            matched_idx = distances.index(min(distances))
            matches.append((c, previous_centroids[matched_idx]))
    return matches

def detect_entry_exit(centroid, entry_line_y, exit_line_y):
    _, y = centroid
    if y < entry_line_y:
        return 'enter'
    elif y > exit_line_y:
        return 'exit'
    return 'none'
