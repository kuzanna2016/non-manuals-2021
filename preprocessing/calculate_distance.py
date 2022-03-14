import pandas as pd

from .distances import find_mean_dist, find_mean_perp_dist, find_mean_perp_plane_dist


def main(open_face_fp, distance_config, save_to):
    of = pd.read_csv(open_face_fp, index_col=[0, 1], sep='\t')
    distance_type = distance_config.get('distance_type')
    inner = distance_config.get('inner_eyebrows_points')
    outer = distance_config.get('outer_eyebrows_points')
    other_points = distance_config.get('other_points')
    if 'perp' in distance_type:
        if 'line' in distance_type:
            df = find_mean_perp_dist(
                of,
                inner_brows=inner,
                outer_brows=outer,
                point_1=other_points[0],
                point_2=other_points[1]
            )
        else:
            df = find_mean_perp_plane_dist(
                of,
                inner_brows=inner,
                outer_brows=outer,
                perp_points=other_points
            )
    else:
        df = find_mean_dist(
            of,
            inner_brows=inner,
            outer_brows=outer,
            point_2=other_points[0]
        )
    df.to_csv(save_to, sep='\t')
