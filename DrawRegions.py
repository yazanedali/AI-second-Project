import matplotlib.pyplot as plt
import math
def generate_grid(feature1_min, feature1_max, feature2_min, feature2_max, step):
  ratio=(feature1_max-feature1_min)/200
  #(10-0)/1000=0.01
  const=math.floor(1/ratio)
  feature1_list = [x/const for x in range(int(feature1_min*const),int( (feature1_max + 1)*const), int(1))]
  feature2_list = [y/const for y in range(int((feature2_min)*const), int((feature2_max +1 )*const), int(1))]
  X_grid = []
  for x in feature1_list:
    for y in feature2_list:
      X_grid.append([[x, y]])
  return X_grid


def predict_on_grid(model, X_grid):
  y_predicted = []
  for point in X_grid:
    # Replace with your model's prediction logic using the point coordinates
    y_predicted.append(model.predict(point))  # Replace with actual prediction
  return y_predicted


def plot_classification_regions(X, y, model, feature1_min, feature1_max, feature2_min, feature2_max, step):
    external=plt
    X_grid = generate_grid(feature1_min, feature1_max, feature2_min, feature2_max, step)
    y_predicted = predict_on_grid(model, X_grid)
    unique_classes = list(set(y))
    fig,ax=plt.subplots()
    # Create a scatter plot of the grid points colored by their predicted class
    ax.scatter([point[0][0] for point in X_grid], [point[0][1] for point in X_grid], c=y_predicted, alpha=0.1, cmap=plt.cm.get_cmap('inferno', len(unique_classes)))

    # Scatter plot the training data with different colors for classes
    scatter = ax.scatter([x[0] for x in X], [x[1] for x in X], c=y,cmap=external.cm.get_cmap('inferno', len(unique_classes)))
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Classification Regions")

    # Create a legend for the classes
    class_labels = [f'Class {i}' for i in unique_classes]
    ax.legend(handles=scatter.legend_elements()[0], labels=class_labels)

    # Label each region with its corresponding class label
    for i, class_label in enumerate(unique_classes):
        # Find the centroid of the points predicted to be in this class
        points_in_class = [X_grid[j] for j, predicted_class in enumerate(y_predicted) if predicted_class == class_label]
        if points_in_class:  # Check if there are any points predicted as this class
            centroid = [sum(x) / len(points_in_class) for x in zip(*points_in_class)]
            ax.text(centroid[0], centroid[1], str(class_label), color=scatter.legend_elements()[1][i].get_facecolor()[0], fontsize=12, ha='center')
    fig.canvas.draw()
    fig.show()
    # fig.wait_window_destroy() 