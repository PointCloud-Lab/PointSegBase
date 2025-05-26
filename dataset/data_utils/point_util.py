import numpy as np

def gen_gaussian_ball(center, radius, size):
    if not isinstance(radius, np.ndarray):
        radius = np.asarray([radius, radius, radius])
    pts = [np.random.normal(loc=center[i], scale=radius[i], size=size) for i in range(center.shape[0])]
    return np.asarray(pts).transpose()


def gen_point_cloud(high, low, center_num, size, scale=1, dim=3):
    normalized_centers = np.random.rand(center_num, dim)
    centers = (high - low) * normalized_centers + low
    ball_pts_ratio = np.random.rand(center_num, )
    ball_pts_ratio = ball_pts_ratio / np.sum(ball_pts_ratio)
    ball_pts_num = (size * ball_pts_ratio).astype(np.int)
    ball_pts_num[-1] = size - np.sum(ball_pts_num[:-1])
    radius_sum = (high - low) * float(scale)
    radius = radius_sum * ball_pts_ratio

    points = []
    for i in range(center_num):
        points.append(gen_gaussian_ball(centers[i], radius[i], ball_pts_num[i]))
    return np.clip(np.vstack(points), low, high)