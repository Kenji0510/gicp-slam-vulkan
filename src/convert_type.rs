use nalgebra::Point3;

use crate::types::PointXYZIT;

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
