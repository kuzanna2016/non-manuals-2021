import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()


def distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=1)
    dist = np.sqrt(squared_dist)
    return dist


def distance_perp_line(points, p1_line, p2_line):
    ap = points - p1_line
    ab = p2_line - p1_line
    try:
        norm = np.linalg.norm(ab)
        cross = np.linalg.norm(np.cross(ab, ap), axis=1)
        distance = cross / norm
    except ZeroDivisionError:
        distance = 0
    return distance


def foot_perp_line(p, p1_line, p2_line):
    ap = p - p1_line
    ab = p2_line - p1_line
    try:
        point = p1_line + np.dot(ap, ab) / np.dot(ab, ab) * ab
    except ZeroDivisionError:
        point = np.array([0, 0])
    return point


def find_plane(p1, p2=None, p3=None, parallel_plane='xz'):
    if p3 is None:
        if p2 is None:
            p2 = p1.copy()
            if parallel_plane[0] == 'z':
                p2[2] = p1[2] + 1
            if parallel_plane[0] == 'y':
                p2[1] = p1[1] + 1
            else:
                p2[0] = p1[0] + 1
        p3 = (p1 + p2) / 2
        if parallel_plane[1] == 'z':
            p3[2] = p3[2] + 1
        if parallel_plane[1] == 'y':
            p3[1] = p3[1] + 1
        else:
            p3[0] = p3[0] + 1

    v1 = p2 - p1
    v2 = p3 - p1

    cp = np.cross(v2, v1)
    a, b, c = cp

    d = np.dot(cp, p3)
    return [a, b, c, d]


def distance_perp_plane(points, plane_points, parallel_plane='xz', rotate_angles=None, dims='XYZ', to_plot=False):
    plane = find_plane(*plane_points, parallel_plane=parallel_plane)
    a, b, c, d = plane
    if to_plot:
        plot_plane([a, b, c, d], plane_points + points.tolist())
    if rotate_angles is not None:
        norm = rotate_plane(np.array([a, b, c]), rotate_angles, dims)
        a, b, c = norm
        d = np.dot(norm, plane_points[0])
        if to_plot:
            plot_plane([a, b, c, d], plane_points + points.tolist())

    coef = np.array([a, b, c])
    norm = np.abs(np.dot(points, coef) - d)
    sqrt = np.sqrt(np.sum(coef ** 2))
    if sqrt == 0:
        d = np.zeros_like(norm)
    else:
        d = norm / sqrt
    return d


def foot_perp_plane(a, b, c, d, x1, y1, z1):
    k = (-a * x1 - b * y1 - c * z1 + d) / (a * a + b * b + c * c)
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    return x2, y2, z2


def rotate_plane(norm_v, angles, dims='XYZ'):
    # rotate to head pose
    rotation = R.from_euler(dims, angles)
    rotated = rotation.apply(norm_v)
    return rotated


def min_max_scale(X, x_min, x_max, axis=0):
    nom = (X - X.min(axis=axis)) * (x_max - x_min)
    denom = X.max(axis=axis) - X.min(axis=axis)
    denom[denom == 0] = 1
    return x_min + nom / denom


def scaled_distance(row, mean=False, scale=True, center=27, perp='brows'):
    Xs = row.filter(regex='^X_\d\d').sort_index()
    Ys = row.filter(regex='^Y_\d\d').sort_index()
    Zs = row.filter(regex='^Z_\d\d').sort_index()

    # rotate face
    rotation = R.from_euler('XYZ', row[['pose_Rx', 'pose_Ry', 'pose_Rz']])

    face = np.stack([Xs['X_00':'X_67'], Ys['Y_00':'Y_67'], Zs['Z_00':'Z_67']], axis=1)
    face_rotated = rotation.apply(face, inverse=True)

    if scale:
        face_rotated_scaled = scale(face_rotated, -100, 100)
    else:
        face_rotated_scaled = face_rotated - face_rotated[center, :]

    # pick up the eyes points
    p1 = face_rotated_scaled[45, :]
    p2 = face_rotated_scaled[36, :]
    p3 = np.mean([p1, p2], axis=0)

    # find coefficients to eyes plane
    a, b, c, d = find_plane(np.array([0, p3[1], 0]), np.array([1, p3[1], 1]), np.array([1, p3[1], 2]), perp=False)

    # finding perpendicular from lower nose to the eyes plane
    foot_eyes_nose = foot(a, b, c, -d, face_rotated_scaled[30, 0], face_rotated_scaled[30, 1],
                          face_rotated_scaled[30, 2])
    d_eyes_nose = distance(face_rotated_scaled[30, :], foot_eyes_nose)

    # finding perpendicular from brow to the eyes plane
    if perp == 'brows':
        perps = []
        for brow in [18, 20, 23, 25]:
            foot_perp = foot_perp_plane(a, b, c, -d,
                                        face_rotated_scaled[brow, 0],
                                        face_rotated_scaled[brow, 1],
                                        face_rotated_scaled[brow, 2])
            d_perp = distance(face_rotated_scaled[brow, :], foot_perp)
            if mean:
                perps.append((d_perp + d_eyes_nose) / 2)
            else:
                perps.append(d_perp)
        return perps

    if perp == 'nose':
        p_nose = face_rotated_scaled[30, :]
        d_nose = np.dot(np.array([a, b, c]), p_nose)
        foot_nose_plane = foot_perp_plane(a, b, c, -d_nose,
                                          face_rotated_scaled[27, 0],
                                          face_rotated_scaled[27, 1],
                                          face_rotated_scaled[27, 2])
        d_nose_plane = distance(face_rotated_scaled[27, :], foot_nose_plane)
        return d_nose_plane


# ======================================================================================


def find_perp(df, brow_points, perp_points, axes='XYZ', dist='line', plot_face=False):
    if len(axes) == 2:
        if len(perp_points) < 2:
            raise ValueError('no point 2 of line perpendicular')
        perp_point_1, perp_point_2 = perp_points[:2]
        column_names = [f'{brow_point}perp{perp_point_1}_{perp_point_2}' for brow_point in brow_points]
        p1 = [[f'{ax}_{brow_point}' for ax in axes] for brow_point in brow_points]
        p2 = [f'{ax}_{perp_point_1}' for ax in axes]
        p3 = [f'{ax}_{perp_point_2}' for ax in axes]
    elif len(axes) == 3:
        if dist == 'line':
            if len(perp_points) < 2:
                raise ValueError('no point 2 of line perpendicular')
            perp_point_1, perp_point_2 = perp_points[:2]
            column_names = [f'{brow_point}perp{perp_point_1}_{perp_point_2}_3d' for brow_point in brow_points]
            p1 = [[f'{ax}_{brow_point}' for ax in axes] for brow_point in brow_points]
            p2 = [f'{ax}_{perp_point_1}' for ax in axes]
            p3 = [f'{ax}_{perp_point_2}' for ax in axes]
        elif dist == 'plane':
            points_str = "_".join(map(str, perp_points))
            column_names = [f'{brow_point}perp_plane{points_str}_3d' for brow_point in brow_points]

            points = [[f'{ax}_{p}' for ax in axes] for n, p in enumerate(perp_points)]
            if plot_face:
                p1 = [[f'{ax}_{point}' for ax in axes] for point in range(47)]
            else:
                p1 = [[f'{ax}_{brow_point}' for ax in axes] for brow_point in brow_points]
            df[column_names] = df.progress_apply(
                lambda row: distance_perp_plane(points=np.array([row[p].to_numpy(dtype='float') for p in p1]),
                                                plane_points=[row[p].to_numpy(dtype='float') for p in points],
                                                rotate_angles=row[['pose_Rx',
                                                                   'pose_Ry',
                                                                   'pose_Rz']].to_numpy(dtype='float'),
                                                to_plot=plot_face
                                                ),
                axis=1,
                result_type='expand')
        else:
            raise ValueError('wrong dist')
    else:
        raise ValueError('wrong number of axes')

    if dist == 'line':
        df[column_names] = df.progress_apply(
            lambda row: distance_perp_line(np.array([row[p].to_numpy(dtype='float') for p in p1]),
                                           row[p2].to_numpy(dtype='float'),
                                           row[p3].to_numpy(dtype='float')),
            axis=1,
            result_type='expand')


def find_foot(df, brow_point, perp_point_1, perp_point_2):
    column_name = f'{brow_point}perp{perp_point_1}_{perp_point_2}_foot_'
    df[[column_name + 'x', column_name + 'y']] = df.apply(lambda row: foot_perp_line(row[[f'x_{brow_point}',
                                                                                          f'y_{brow_point}']].to_numpy(
        dtype='float'),
                                                                                     row[[f'x_{perp_point_1}',
                                                                                          f'y_{perp_point_1}']].to_numpy(
                                                                                         dtype='float'),
                                                                                     row[[f'x_{perp_point_2}',
                                                                                          f'y_{perp_point_2}']].to_numpy(
                                                                                         dtype='float')),
                                                          axis=1,
                                                          result_type='expand')


def find_dist(df, points_1, point_2, dim=2):
    if dim == 2:
        column_names = [f'dist{point_1}_{point_2}' for point_1 in points_1]
        p2 = [f'x_{point_2}', f'y_{point_2}']
        p1 = [[f'x_{point_1}', f'y_{point_1}'] for point_1 in points_1]
    elif dim == 3:
        column_names = [f'dist{point_1}_{point_2}_3d' for point_1 in points_1]
        p2 = [f'X_{point_2}', f'Y_{point_2}', f'Z_{point_2}']
        p1 = [[f'X_{point_1}', f'Y_{point_1}', f'Z_{point_1}'] for point_1 in points_1]
    else:
        raise ValueError('wrong dimension')

    df[column_names] = df.progress_apply(lambda row: distance(np.array([row[p].to_numpy(dtype='float') for p in p1]),
                                                              row[p2].to_numpy(dtype='float')),
                                         axis=1,
                                         result_type='expand')


def find_mean_dist(df, inner_brows, outer_brows, point_2, dim=3, override=False):
    l_brow, r_brow = inner_brows
    column_name = f'inner_dist{point_2}_3d'
    if column_name not in df.columns or override:
        find_dist(df, inner_brows + outer_brows, point_2, dim=dim)
        df[column_name] = df[[f'dist{l_brow}_{point_2}_3d', f'dist{r_brow}_{point_2}_3d']].mean(axis=1)

    l_brow, r_brow = outer_brows
    column_name = f'outer_dist{point_2}_3d'
    if column_name not in df.columns or override:
        df[column_name] = df[[f'dist{l_brow}_{point_2}_3d', f'dist{r_brow}_{point_2}_3d']].mean(axis=1)
    return df


def find_mean_perp_dist(df, inner_brows, outer_brows, point_1, point_2, override=False):
    l_brow, r_brow = inner_brows
    column_name = f'inner_perp_dist{point_1}_{point_2}_3d'
    if column_name not in df.columns or override:
        find_perp(df,
                  brow_points=inner_brows + outer_brows,
                  perp_points=[point_1, point_2],
                  dist='line')
        df[column_name] = df[[f'{l_brow}perp{point_1}_{point_2}_3d',
                              f'{r_brow}perp{point_1}_{point_2}_3d']].mean(axis=1)

    l_brow, r_brow = outer_brows
    column_name = f'outer_perp_dist{point_1}_{point_2}_3d'
    if column_name not in df.columns or override:
        df[column_name] = df[[f'{l_brow}perp{point_1}_{point_2}_3d',
                              f'{r_brow}perp{point_1}_{point_2}_3d']].mean(axis=1)
    return df


def find_mean_perp_plane_dist(df, inner_brows, outer_brows, perp_points, override=False, plot_face=False):
    points_str = "_".join(map(str, perp_points))

    l_brow, r_brow = inner_brows
    column_name = f'inner_perp_plane_dist{points_str}_3d'

    if column_name not in df.columns or override:
        find_perp(df,
                  brow_points=inner_brows + outer_brows,
                  perp_points=perp_points,
                  dist='plane',
                  plot_face=plot_face)
        df[column_name] = df[[f'{l_brow}perp_plane{points_str}_3d',
                              f'{r_brow}perp_plane{points_str}_3d']].mean(axis=1)

    l_brow, r_brow = outer_brows
    column_name = f'outer_perp_plane_dist{points_str}_3d'
    if column_name not in df.columns or override:
        df[column_name] = df[[f'{l_brow}perp_plane{points_str}_3d',
                              f'{r_brow}perp_plane{points_str}_3d']].mean(axis=1)
    return df


def plot_plane(plane, points, fp=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the original points. We use zip to get 1D lists of x, y and z
    # coordinates.
    ax.plot(*zip(*points), color='r', linestyle=' ', marker='o')

    grid = []
    if 0 in plane[:-1]:
        parallel_axis = 0
        for i, axis in enumerate(plane[:-1]):
            if axis != 0:
                parallel_axis = i
            else:
                lims = ax.get_w_lims()
                grid.append(np.linspace(*lims[i * 2:i * 2 + 2], 10))
        if len(grid) == 2:
            grid = np.meshgrid(*grid)
        elif len(grid) > 2:
            grid = np.meshgrid(*grid[:2])
        c = plane.pop(parallel_axis)
        a, b, d = plane
        grid.insert(parallel_axis, (d - a * grid[0] - b * grid[1]) / c)
    else:
        x = np.linspace(*ax.get_xlim(), 10)
        z = np.linspace(*ax.get_zlim(), 10)
        grid = np.meshgrid(x, z)
        a, b, c, d = plane
        grid.insert(1, (d - a * grid[0] - c * grid[1]) / b)

    # plot the mesh. Each array is 2D, so we flatten them to 1D arrays
    lims = ax.get_w_lims()
    ax.plot_surface(*grid, alpha=0.2)
    ax.set_xlim(*lims[:2])
    ax.set_ylim(*lims[2:4])
    ax.set_zlim(*lims[4:])

    # plot lines
    ax.plot(*zip(points[28], points[19]), label='dist27')
    ax.plot(*zip(points[40], points[43]))
    perp = foot_perp_line(np.array(points[24]), np.array(points[40]), np.array(points[43]))
    ax.plot(*zip(points[24], perp), label='perp_dist39_42')
    perp_plane = foot_perp_plane(a, b, c, d, *points[26])
    ax.plot(*zip(points[26], perp_plane), label='perp_plane_dist27')

    # adjust the view so we can see the point/plane alignment
    plt.tight_layout()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if fp is not None:
        plt.savefig(fp)
    plt.show()
