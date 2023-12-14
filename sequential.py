from utils import *
import numpy as np
import matplotlib.pyplot as plt

def twoPass(image):

    rows, cols = len(image), len(image[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    current_label = 0
    equivalences = {}

    # First pass - assign labels
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 1:
                neighbors = [
                    labels[i-1][j] if i > 0 else 0,  # Top neighbor
                    labels[i][j-1] if j > 0 else 0  # Left neighbor
                ]
                if all(neighbor == 0 for neighbor in neighbors):
                    current_label += 1
                    labels[i][j] = current_label
                else:
                    non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]
                    if len(non_zero_neighbors) == 1:
                        labels[i][j] = non_zero_neighbors[0]
                    else:
                        min_neighbor = min(non_zero_neighbors)
                        labels[i][j] = min_neighbor
                        for neighbor in non_zero_neighbors:
                            if neighbor != min_neighbor:
                                if min_neighbor not in equivalences:
                                    equivalences[min_neighbor] = set()
                                equivalences[min_neighbor].add(neighbor)

    changed = True
    while changed:
        changed = False
        items = list(equivalences.items())
        for i, (key1, set1) in enumerate(items):
            for key2, set2 in items[i + 1:]:
                # print(key1, key2)
                if key1 in set2 or key2 in set1 or not set1.isdisjoint(set2):
                    if key1 < key2:
                        set1.update(set2)
                        equivalences[key1] = set1
                        equivalences[key1].add(key2)
                        del equivalences[key2]
                        changed = True
                        break
                    else:
                        set2.update(set1)
                        equivalences[key2] = set2
                        equivalences[key2].add(key1)
                        del equivalences[key1]
                        changed = True
                        break
            if changed:
                break

    # [print(row) for row in labels]
    # print(equivalences)

    # Second pass - resolve equivalences
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                label = labels[i][j]
                for key, values in equivalences.items():
                    if label in values:
                        label = key
                        break
                while label in equivalences:
                    # print(label, equivalences[label])
                    label = min(equivalences[label])
                labels[i][j] = label

    return labels


if __name__ == "__main__":
    # binary_image = binariseImage("flower.jpg")
    # m, n = 192, 144
    m = n = 512
    binary_image = generateImage(m, n, 42)
    # for row in binary_image:
    #     print(row)

    # print("\n")

    result = twoPass(binary_image)
    # [print(row) for row in result]

    # plt.imshow(result, cmap='Blues')
    # plt.show()
