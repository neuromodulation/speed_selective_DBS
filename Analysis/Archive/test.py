import matplotlib.pyplot as plt
import numpy as np

# Generate example data for multiple plots
np.random.seed(0)
data = []
for _ in range(5):
    x = np.random.rand(50)
    y = np.random.rand(50)
    data.append((x, y))

# List to store selected points for each plot
all_selected_points = []

for idx, (x, y) in enumerate(data):
    # Create a scatter plot for each set of data
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)

    # List to store selected point indices for current plot
    selected_points = []

    # Function to handle click events for current plot
    def on_click(event):
        if event.button == 1:  # Check for left mouse button click
            if scatter.contains(event)[0]:
                ind = scatter.contains(event)[1]["ind"][0]
                selected_points.append(ind)
                print(f"Selected point in plot {idx}: {ind}")

    # Connect the button press event to the figure for current plot
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    # Append the selected points for the current plot to the list
    all_selected_points.append(selected_points)

# Print the indices of selected points for all plots
print("Indices of selected points for all plots:", all_selected_points)