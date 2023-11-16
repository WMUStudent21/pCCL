def generate_image(n=10, m=10, SEED=42):
    
    import random

    random.seed(SEED)

    return [[(random.choice([0, 1])) for _ in range(m)] for _ in range(n)]

def binarise_image(image_path):

    import numpy as np
    from PIL import Image

    return np.where(np.array(Image.open(image_path).convert('L')) > 128, 0, 1)