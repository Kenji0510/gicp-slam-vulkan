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
