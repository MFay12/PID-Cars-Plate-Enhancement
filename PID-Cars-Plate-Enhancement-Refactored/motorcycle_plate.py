from heuristic_detector import HeuristicPlateDetector, draw_candidates

_DETECTOR = HeuristicPlateDetector()

def search_motorcycle_plate(original_img, color_img):
    """Searches for motorcycle plates keeping the original PID contract."""
    top_5 = _DETECTOR.search(original_img, families=("motorcycle",), limit=5)
    debug_img = draw_candidates(color_img, top_5, color=(255, 165, 0))
    return debug_img, top_5
