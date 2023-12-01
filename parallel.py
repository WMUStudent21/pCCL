'''
Ref: https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Kun-Lin-Spring-2020.pdf
'''

import utils
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def compute_global_equivalences(labels):
    """
    Compute global equivalences based on neighboring row results.
    """
    global_equivalences = {}
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] != 0:
                neighbors = [labels[i-1][j] if i > 0 else 0, labels[i][j-1] if j > 0 else 0]
                if not (all(neighbor == 0 for neighbor in neighbors)):
                    non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]
                    min_neighbor = min(non_zero_neighbors)
                    labels[i][j] = min_neighbor
                    for neighbor in non_zero_neighbors:
                        if neighbor != min_neighbor:
                            global_equivalences[neighbor] = min_neighbor

    return global_equivalences


def twoPassParallel(image, comm):

    def FirstPass(labels, start_row, end_row, cols):
        current_label = 0
        
        # First pass - assign labels locally
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
        return labels

    # def SecondPass(labels, global_equivalences, start_row, end_row, cols):
    #     # Second pass - resolve equivalences locally
    #     for i in range(start_row, end_row):
    #         for j in range(cols):
    #             if labels[i][j] != 0:
    #                 label = labels[i][j]
    #                 while label in global_equivalences:
    #                     label = global_equivalences[label]
    #                 labels[i][j] = label
    #     return labels

    rank = comm.Get_rank()
    size = comm.Get_size()

    # print(rank,size)

    rows, cols = len(image), len(image[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]

    # Split the rows among processes
    local_rows = rows // size
    start_row = rank * local_rows
    end_row = start_row + local_rows if rank < size - 1 else rows

    labels = FirstPass(labels, start_row, end_row, cols)

    # Send locals to master to be combined, needs to block    need it, has 75% speedup
    comm.Barrier()

 

    if rank == 0:
           # Master process combines results and computes global equivalences
        global_equivalences = {}
        gathered_results = comm.gather(labels[start_row:end_row], root=0)
        gathered_results = np.concatenate(gathered_results, axis=0)
        # Compute global equivalences based on neighboring row results
        global_equivalences = compute_global_equivalences(gathered_results)

    # Broadcast the global equivalent list to all processors
    # global_equivalences = comm.bcast(global_equivalences, root=0)

    # Using global equivalence, relabel each row
    # comm.Barrier()

    # labels = SecondPass(labels, global_equivalences, start_row, end_row, cols)

    # print(rank)
    # print(labels)

    # gathered_results = comm.Scatter([labels[start_row:end_row]], root=0, out=None)
    # gathered_results = np.concatenate(gathered_results, axis=0)
    # print(gathered_results)

    return gathered_results, global_equivalences
    # return gathered_results


def SecondPass(gathered_labels, global_equivalences, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # print(rank, size)

    rows, cols = len(gathered_labels), len(gathered_labels[0])

    # Split the rows among processes
    local_rows = rows // size
    start_row = rank * local_rows
    end_row = start_row + local_rows if rank < size - 1 else rows

    if rank != 0:
        # Second pass - resolve equivalences locally
        for i in range(start_row, end_row):
            for j in range(cols):
                print(gathered_labels[i])
                if gathered_labels[i][j] != 0:
                    label = gathered_labels[i][j]
                    print(label, gathered_labels[i][j])
                    while label in global_equivalences:
                        label = global_equivalences[label]
                    gathered_labels[i][j] = label

    return labels

if __name__ == "__main__":
    # binary_image = utils.binarise_image("flower.jpg")
    binary_image = utils.generate_image(7, 7, 42)

    comm = MPI.COMM_WORLD
    labels, equivalences = twoPassParallel(binary_image, comm)

    if comm.Get_rank != 0:
        result = SecondPass(labels, equivalences, comm)
    MPI.Finalize()

    print(result)

    # if comm.Get_rank() == 0:
    #     plt.imshow(result, cmap='Blues')
    #     plt.show()
    # MPI.Finalize()


