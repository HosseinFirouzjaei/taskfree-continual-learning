import numpy as np


def make_quadrant_data(n_points, quadrant, seed=0):
    """
    Make 4D points that are inside or outside the unit sphere.

    n_points: how many points to make
    quadrant: 1 = all coordinates positive, 2 = all coordinates negative
    seed: a fixed number, so we get the same points every run
    """
    rng = np.random.default_rng(seed)

    half = n_points // 2
    inside_points = []
    outside_points = []

    # Make random points until both piles are full.
    while len(inside_points) < half or len(outside_points) < half:
        p = rng.uniform(0, 1, size=4)          # one point, 4 numbers between 0 and 1
        dist = np.sqrt(np.sum(p ** 2))         # distance from the center (Pythagoras in 4D)
        if dist < 1 and len(inside_points) < half:
            inside_points.append(p)            # inside the sphere
        elif dist >= 1 and len(outside_points) < half:
            outside_points.append(p)           # outside the sphere

    points = np.array(inside_points + outside_points)
    labels = np.array([1] * half + [0] * half)  # 1 = inside, 0 = outside

    if quadrant == 2:
        points = -points                       # flip signs to the other quadrant

    return points, labels


if __name__ == "__main__":
    X, y = make_quadrant_data(2000, quadrant=1)
    print("points shape:", X.shape)
    print("labels shape:", y.shape)
    print("inside count:", int(y.sum()))
    print("outside count:", int(len(y) - y.sum()))
    print("first point:", X[0])
    print("first label:", y[0])