def generate_image(n=10, m=10, SEED=42):
    import random
    random.seed(SEED)
    return [[(random.choice([0, 1])) for _ in range(m)] for _ in range(n)]