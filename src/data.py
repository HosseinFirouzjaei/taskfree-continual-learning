import numpy as np


def make_quadrant_data(n_points, quadrant, seed=0, margin=0.2):
    """
    Make 4D points that are clearly inside or outside the unit sphere.

    n_points : how many points to make (half inside, half outside)
    quadrant : 1 = all coordinates positive, 2 = all coordinates negative
    seed     : a fixed number, so we get the same points every run
    margin   : a gap left empty around the sphere, so the two classes
               are clearly separable (no ambiguous points on the boundary)
    """
    rng = np.random.default_rng(seed)
    half = n_points // 2
    inside_points = []
    outside_points = []

    # Keep making random points until both piles are full.
    while len(inside_points) < half or len(outside_points) < half:
        p = rng.uniform(0, 1, size=4)          # one point, 4 numbers in [0, 1]
        dist = np.sqrt(np.sum(p ** 2))         # distance from the centre
        if dist < 1 - margin and len(inside_points) < half:
            inside_points.append(p)            # clearly inside
        elif dist > 1 + margin and len(outside_points) < half:
            outside_points.append(p)           # clearly outside

    points = np.array(inside_points + outside_points)
    labels = np.array([1] * half + [0] * half)  # 1 = inside, 0 = outside

    if quadrant == 2:
        points = -points                        # flip to the other quadrant
    return points, labels


if __name__ == "__main__":
    X, y = make_quadrant_data(2000, quadrant=1)
    print("points shape:", X.shape)
    print("inside count:", int(y.sum()), " outside count:", int(len(y) - y.sum()))
