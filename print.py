def print_grid_world(rows=5, cols=5):
    """
    Print a grid world of size rows x cols,
    where each cell is labeled with its (row, column) coordinates.
    """
    for row in range(rows):
        # Build a list of coordinate labels for this row
        labels = [f"({row},{col})" for col in range(cols)]
        # Join them with spaces and print
        print(' '.join(labels))

if __name__ == "__main__":
    print_grid_world()
