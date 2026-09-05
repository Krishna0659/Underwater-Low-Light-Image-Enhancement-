import cv2
import numpy as np


def sequential_labeling(binary_image):
    """
    Two-pass Sequential Labeling Algorithm
    using 4-connectivity.

    Input:
        binary_image : 2D numpy array
                       Background = 0
                       Foreground = non-zero

    Output:
        labeled_image : 2D numpy array
    """

    # Make sure image is binary
    binary = (binary_image > 0).astype(np.uint8)

    rows, cols = binary.shape

    # Output label image
    labels = np.zeros((rows, cols), dtype=np.int32)

    # Label counter
    next_label = 1

    # Equivalence table
    parent = {}

    # ----------------------------------------
    # FIND function (Union-Find)
    # ----------------------------------------
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # ----------------------------------------
    # UNION function
    # ----------------------------------------
    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # ========================================
    # PASS 1
    # ========================================

    for i in range(rows):
        for j in range(cols):

            # Background pixel
            if binary[i, j] == 0:
                continue

            # Get upper and left neighbors
            top = labels[i - 1, j] if i > 0 else 0
            left = labels[i, j - 1] if j > 0 else 0

            # --------------------------------
            # CASE 1:
            # Both neighbors are background
            # --------------------------------
            if top == 0 and left == 0:

                labels[i, j] = next_label

                # Initialize equivalence
                parent[next_label] = next_label

                next_label += 1

            # --------------------------------
            # CASE 2:
            # Only top is labeled
            # --------------------------------
            elif top != 0 and left == 0:

                labels[i, j] = top

            # --------------------------------
            # CASE 3:
            # Only left is labeled
            # --------------------------------
            elif top == 0 and left != 0:

                labels[i, j] = left

            # --------------------------------
            # CASE 4:
            # Both have same label
            # --------------------------------
            elif top == left:

                labels[i, j] = top

            # --------------------------------
            # CASE 5:
            # Both have different labels
            # --------------------------------
            else:

                # Assign one of the labels
                labels[i, j] = min(top, left)

                # Record equivalence
                union(top, left)

    # ========================================
    # PASS 2
    # ========================================

    # Create final label mapping
    final_labels = {}
    new_label = 1

    for i in range(rows):
        for j in range(cols):

            if labels[i, j] != 0:

                root = find(labels[i, j])

                # Give each connected component
                # a clean sequential label
                if root not in final_labels:
                    final_labels[root] = new_label
                    new_label += 1

                labels[i, j] = final_labels[root]

    return labels


# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    binary_image = np.array([
        [0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0]
    ], dtype=np.uint8)

    labeled = sequential_labeling(binary_image)

    print("Binary Image:")
    print(binary_image)

    print("\nLabeled Image:")
    print(labeled)