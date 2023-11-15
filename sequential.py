def label_components(image):
    rows, cols = len(image), len(image[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    current_label = 0
    equivalences = {}

    # First pass

    # parallelise the row-major
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 1:
                neighbors = [labels[i-1][j] if i > 0 else 0, labels[i][j-1] if j > 0 else 0]
                if all(neighbor == 0 for neighbor in neighbors):
                    current_label += 1
                    labels[i][j] = current_label
                else:
                    non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]
                    min_neighbor = min(non_zero_neighbors)
                    labels[i][j] = min_neighbor
                    for neighbor in non_zero_neighbors:
                        if neighbor != min_neighbor:
                            equivalences[neighbor] = min_neighbor

    # Second pass - resolve equivalences
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                label = labels[i][j]
                while label in equivalences:
                    label = equivalences[label]
                labels[i][j] = label

    return labels


import random
# Example usage:
binary_image = [[(random.choice([0, 1])) for _ in range(8)] for _ in range(8)]
for row in binary_image:
    print(row)

print("\n")

result = label_components(binary_image)
for row in result:
    print(row)
