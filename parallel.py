'''
Ref: https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Kun-Lin-Spring-2020.pdf
'''

import utils
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def twoPassParallel(image):
    rows, cols = len(image), len(image[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    current_label = 0
    equivalences = {}

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # First pass - assign labels

    chunk_size = rows // size
    start_row = rank * chunk_size
    end_row = start_row + chunk_size if rank < size - 1 else rows

    print(start_row, end_row)

    # Parallelize Row-Major
    for i in range(start_row, end_row):
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

    comm.Barrier()
    # print(f"Rank {rank} results after first pass")


    # Second pass - resolve equivalences
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                label = labels[i][j]
                while label in equivalences:
                    label = equivalences[label]
                labels[i][j] = label

    return labels


if __name__ == "__main__":
    binary_image = utils.binarise_image("flower.jpg")
    # binary_image = utils.generate_image(10, 10, 42)
    # for row in binary_image:
        # print(row)

    # print("\n")

    result = twoPassParallel(binary_image)
    # for row in result:
        # print(row)

    plt.imshow(result, cmap='Blues')
    plt.show()
