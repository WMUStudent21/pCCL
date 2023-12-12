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

    # First pass - assign labels
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] != 0:
                neighbors = [
                    labels[i-1][j] if i > 0 else 0,  # Top neighbor
                    labels[i+1][j] if i < len(labels) - 1 else 0,  # Bottom neighbor
                    labels[i][j-1] if j > 0 else 0,  # Left neighbor
                    labels[i][j+1] if j < len(labels[0]) - 1 else 0  # Right neighbor
                ]
                non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]
                if non_zero_neighbors:
                    min_neighbor = min(non_zero_neighbors)
                    for neighbor in non_zero_neighbors:
                        if neighbor != min_neighbor:
                            if min_neighbor not in global_equivalences:
                                global_equivalences[min_neighbor] = set()
                            global_equivalences[min_neighbor].add(neighbor)

    changed = True
    while changed:
        changed = False
        items = list(global_equivalences.items())
        for i, (key1, set1) in enumerate(items):
            for key2, set2 in items[i + 1:]:
                # print(key1, key2)
                if key1 in set2 or key2 in set1 or not set1.isdisjoint(set2):
                    if key1 < key2:
                        set1.update(set2)
                        global_equivalences[key1] = set1
                        global_equivalences[key1].add(key2)
                        del global_equivalences[key2]
                        changed = True
                        break
                    else:
                        set2.update(set1)
                        global_equivalences[key2] = set2
                        global_equivalences[key2].add(key1)
                        del global_equivalences[key1]
                        changed = True
                        break
            if changed:
                break

    return global_equivalences

def twoPassParallel(image, comm, size):

    def FirstPass(labels, start_row, end_row, cols):
        current_label = comm.Get_rank() * (rows // size) + end_row
        equivalences = {}

        # First pass - assign labels
        for i in range(start_row, end_row):
            for j in range(cols):
                if image[i][j] == 1:
                    neighbors = [
                        labels[i-1][j] if i > 0 else 0,  # Top neighbor
                        labels[i+1][j] if i < len(labels) - 1 else 0,  # Bottom neighbor
                        labels[i][j-1] if j > 0 else 0,  # Left neighbor
                        labels[i][j+1] if j < len(labels[0]) - 1 else 0  # Right neighbor
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


        return labels, equivalences

    rank = comm.Get_rank()
    rows, cols = len(image), len(image[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]

    if rank != 0:

        # Split the rows among processes
        local_rows = rows // size
        start_row = (rank-1) * local_rows
        end_row = start_row + local_rows if rank < size else rows
        # print(f"Total rows {rows}\nLocal rows {local_rows,start_row, end_row} for rank {rank}")
        labels, equivalences = FirstPass(labels, start_row, end_row, cols)
        # print("\n".join(map(lambda row: ' '.join(map(str, row)), labels)))

        comm.send((start_row, end_row, labels), dest=0)
        # print(f"Rank {rank} sent {start_row, end_row} to master")

    return labels

def SecondPass(labels, comm, size):

    rank = comm.Get_rank()
    rows, cols = len(labels), len(labels[0])

    # print(rank)
    # print("\n".join(map(lambda row: ' '.join(map(str, row)), labels)))

    if rank != 0:

        # Split the rows among processes
        local_rows = rows // size
        start_row = (rank-1) * local_rows
        end_row = start_row + local_rows if rank < size else rows

        # Second pass - resolve equivalences locally
        for i in range(start_row, end_row):
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
        
        comm.send((start_row, end_row, labels), dest=0)

    return labels

if __name__ == "__main__":

    # binary_image = utils.binariseImage("flower.jpg")
    # m, n = len(binary_image), len(binary_image[0])
    m = n = 256
    binary_image = utils.generateImage(m, n, 42)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()-1

    labels = twoPassParallel(binary_image, comm, size)

    gathered_labels = [[0]*n for _ in range(m)]
    if comm.Get_rank() == 0:
        for _ in range(1, size+1):
            start, end, data = comm.recv()
            # print(f"Master received {start, end} from rank {_}")
            # print("\n".join(map(lambda row: ' '.join(map(str, row)), data)))
            for i in range(start, end):
                gathered_labels[i] = data[i]
        # print("\n".join(map(lambda row: ' '.join(map(str, row)), gathered_labels)))
        # [print(row) for row in np.array(gathered_labels)]

    global_equivalences = compute_global_equivalences(gathered_labels)
    equivalences = comm.bcast(global_equivalences, root=0)

    if comm.Get_rank() != 0:
        result = SecondPass(labels, comm, size)

    if comm.Get_rank() == 0:
        for _ in range(1, size+1):
            start, end, data = comm.recv()
            for i in range(start, end):
                gathered_labels[i] = data[i]
        # [print(row) for row in np.array(gathered_labels)]
        # plt.imshow(gathered_labels, cmap='Blues')
        # plt.show()
    
    MPI.Finalize()



