use nalgebra::Point3;

use crate::types::{PointXYZCov, PointXYZIT};

pub fn convert_pcd_to_xyz(points: &[PointXYZIT]) -> Vec<Point3<f32>> {
    points.iter().map(|p| Point3::new(p.x, p.y, p.z)).collect()
}

pub fn convert_xyz_to_pcd(points: &[Point3<f32>]) -> Vec<PointXYZIT> {
    points
        .iter()
        .map(|p| PointXYZIT {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: 0.0,
            timestamp: 0.0,
        })
        .collect()
}

pub fn convert_xyz_to_vec(points: &[PointXYZIT]) -> Vec<[f32; 3]> {
    points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

pub fn convert_vec_to_xyz(points: &[[f32; 3]]) -> Vec<PointXYZIT> {
    points
        .iter()
        .map(|p| PointXYZIT {
            x: p[0],
            y: p[1],
            z: p[2],
            intensity: 0.0,
            timestamp: 0.0,
        })
        .collect()
}

pub fn convert_point3_to_vec(points: &[Point3<f32>]) -> Vec<[f32; 3]> {
    points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

pub fn convert_vec_to_point3(points: &[[f32; 3]]) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|p| Point3::new(p[0], p[1], p[2]))
        .collect()
}

pub fn convert_vec_point_cov_to_pcd_xyzcov(
    points: &[[f32; 3]],
    cov: &[[f32; 9]],
) -> Vec<PointXYZCov> {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| PointXYZCov {
            x: p[0],
            y: p[1],
            z: p[2],
            cov_xx: cov[i][0],
            cov_xy: cov[i][1],
            cov_xz: cov[i][2],
            cov_yy: cov[i][4],
            cov_yz: cov[i][5],
            cov_zz: cov[i][8],
        })
        .collect()
}
