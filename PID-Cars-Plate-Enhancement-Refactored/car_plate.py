from heuristic_detector import HeuristicPlateDetector, draw_candidates

_DETECTOR = HeuristicPlateDetector()

def search_car_plate(original_img, color_img):
    """Searches for horizontal plates keeping the original PID contract."""
    top_5 = _DETECTOR.search(original_img, families=("car",), limit=5)
    debug_img = draw_candidates(color_img, top_5, color=(0, 165, 255))
    return debug_img, top_5
