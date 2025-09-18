import numpy as np

def detect_primitives(image: np.ndarray) -> list[dict]:
    """
    Analyzes an image and extracts geometric primitives.
    
    Args:
        image: A NumPy array representing the input image.
        
    Returns:
        A list of dictionaries, where each dictionary represents a
        geometric primitive (e.g., line, circle).
    """
    print("vision.py: Detecting primitives...")
    
    # --- ML Engineer 1 (CV Lead) replaces this mock data ---
    # This mock data unblocks the other engineers immediately.
    mock_primitives = [
        {'type': 'line', 'start': (10, 10), 'end': (100, 10)},
        {'type': 'line', 'start': (100, 10), 'end': (100, 100)},
        {'type': 'line', 'start': (100, 100), 'end': (10, 100)},
        {'type': 'line', 'start': (10, 100), 'end': (10, 10)},
        {'type': 'circle', 'center': (55, 55), 'radius': 20},
    ]
    # ---------------------------------------------------------
    
    return mock_primitives
