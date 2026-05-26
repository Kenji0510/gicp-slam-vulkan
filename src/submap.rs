use nalgebra::{Isometry3, Matrix4, Point3, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};


pub type SubmapId = u64;
pub type FrameId = u64;

#[derive(Debug, Clone)]
pub struct SubmapConfig {
    pub max_frames_per_submap: usize,
    pub max_distance_per_submap: f32,
    pub max_points_per_submap: usize,
}

#[derive(Debug, Clone)]
pub struct Submap {
    pub id: SubmapId,
    pub start_frame_id: FrameId,
    pub end_frame_id: FrameId,

    pub points_local_reg: Vec<Point3<f32>>,

    pub points_local_map: Vec<Point3<f32>>,

    pub pose_world: Isometry3<f64>,

    pub odom_pose_world: Isometry3<f64>,

    pub center_world: Point3<f32>,
    pub radius: f32,

    pub stats: SubmapStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmapStats {
    pub num_frames: usize,
    pub num_points_reg: usize,
    pub num_points_map: usize,
    pub travel_distance: f32,
}

pub struct SubmapBuilder {
    id: SubmapId,
    start_frame_id: FrameId,
    last_frame_id: FrameId,

    /// submap local -> world
    anchor_pose_world: Isometry3<f64>,

    points_local_reg: Vec<Point3<f32>>,
    points_local_map: Vec<Point3<f32>>,

    frame_count: usize,
    accumulated_distance: f32,
    last_origin_world: Option<Point3<f32>>,
}

impl SubmapBuilder {
    pub fn new(
        id: SubmapId,
        start_frame_id: FrameId,
        anchor_pose_world: Isometry3<f64>,
    ) -> Self {
        Self {
            id,
            start_frame_id,
            last_frame_id: start_frame_id,
            anchor_pose_world,
            points_local_reg: Vec::new(),
            points_local_map: Vec::new(),
            frame_count: 0,
            accumulated_distance: 0.0,
            last_origin_world: None,
        }
    }

    pub fn insert_frame_points(
        &mut self,
        frame_id: FrameId,
        frame_pose_world: &Isometry3<f64>,
        points_lidar_reg: &[Point3<f32>],
        points_lidar_map: &[Point3<f32>],
    ) {
        self.last_frame_id = frame_id;
        self.frame_count += 1;

        let anchor_inv = self.anchor_pose_world.inverse();

        // LiDAR local -> world -> submap local
        for p in points_lidar_reg {
            let p_lidar = p.cast::<f64>();
            let p_world = frame_pose_world.transform_point(&p_lidar);
            let p_submap = anchor_inv.transform_point(&p_world);
            self.points_local_reg.push(p_submap.cast::<f32>());
        }

        for p in points_lidar_map {
            let p_lidar = p.cast::<f64>();
            let p_world = frame_pose_world.transform_point(&p_lidar);
            let p_submap = anchor_inv.transform_point(&p_world);
            self.points_local_map.push(p_submap.cast::<f32>());
        }

        let origin_world = Point3::new(
            frame_pose_world.translation.vector.x as f32,
            frame_pose_world.translation.vector.y as f32,
            frame_pose_world.translation.vector.z as f32,
        );

        if let Some(prev) = self.last_origin_world {
            self.accumulated_distance += (origin_world - prev).norm();
        }

        self.last_origin_world = Some(origin_world);
    }

    pub fn should_finish(&self, config: &SubmapConfig) -> bool {
        self.frame_count >= config.max_frames_per_submap
            || self.accumulated_distance >= config.max_distance_per_submap
            || self.points_local_reg.len() >= config.max_points_per_submap
    }

    pub fn finish(self) -> Submap {
        let center_world = Point3::new(
            self.anchor_pose_world.translation.vector.x as f32,
            self.anchor_pose_world.translation.vector.y as f32,
            self.anchor_pose_world.translation.vector.z as f32,
        );

        let radius = self
            .points_local_reg
            .iter()
            .map(|p| p.coords.norm())
            .fold(0.0_f32, f32::max);

        let num_points_reg = self.points_local_reg.len();
        let num_points_map = self.points_local_map.len();

        Submap {
            id: self.id,
            start_frame_id: self.start_frame_id,
            end_frame_id: self.last_frame_id,
            points_local_reg: self.points_local_reg,
            points_local_map: self.points_local_map,
            pose_world: self.anchor_pose_world,
            odom_pose_world: self.anchor_pose_world,
            center_world,
            radius,
            stats: SubmapStats {
                num_frames: self.frame_count,
                num_points_reg,
                num_points_map,
                travel_distance: self.accumulated_distance,
            },
        }
    }
}

pub struct SubmapManager {
    pub config: SubmapConfig,
    pub submaps: Vec<Submap>,
    active: Option<SubmapBuilder>,
    next_submap_id: SubmapId,
}

impl SubmapManager {
    pub fn new(config: SubmapConfig) -> Self {
        Self {
            config,
            submaps: Vec::new(),
            active: None,
            next_submap_id: 0,
        }
    }

    pub fn insert_frame(
        &mut self,
        frame_id: FrameId,
        frame_pose_world: Isometry3<f64>,
        points_lidar_reg: &[Point3<f32>],
        points_lidar_map: &[Point3<f32>],
    ) -> Option<SubmapId> {
        if self.active.is_none() {
            let id = self.next_submap_id;
            self.next_submap_id += 1;

            self.active = Some(SubmapBuilder::new(
                id,
                frame_id,
                frame_pose_world,
            ));
        }

        let active = self.active.as_mut().unwrap();

        active.insert_frame_points(
            frame_id,
            &frame_pose_world,
            points_lidar_reg,
            points_lidar_map,
        );

        if active.should_finish(&self.config) {
            let builder = self.active.take().unwrap();
            let submap = builder.finish();
            let id = submap.id;

            log::info!(
                "Finished submap {}: frames={}..{}, points_reg={}, travel={:.2}m",
                submap.id,
                submap.start_frame_id,
                submap.end_frame_id,
                submap.stats.num_points_reg,
                submap.stats.travel_distance,
            );

            self.submaps.push(submap);
            return Some(id);
        }

        None
    }

    pub fn finalize_active(&mut self) -> Option<SubmapId> {
        let builder = self.active.take()?;
        let submap = builder.finish();
        let id = submap.id;

        log::info!(
            "Finalized last submap {}: frames={}..{}, points_reg={}",
            submap.id,
            submap.start_frame_id,
            submap.end_frame_id,
            submap.stats.num_points_reg,
        );

        self.submaps.push(submap);
        Some(id)
    }

    pub fn get(&self, id: SubmapId) -> Option<&Submap> {
        self.submaps.iter().find(|s| s.id == id)
    }

    pub fn len(&self) -> usize {
        self.submaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.submaps.is_empty()
    }
}

pub fn matrix4_to_isometry3(m: &Matrix4<f64>) -> Isometry3<f64> {
    let rot = m.fixed_view::<3, 3>(0, 0).into_owned();
    let trans = Translation3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);

    let q = UnitQuaternion::from_matrix(&rot);

    Isometry3::from_parts(trans, q)
}

pub fn transform_submap_points_to_world(submap: &Submap) -> Vec<Point3<f32>> {
    submap
        .points_local_map
        .iter()
        .map(|p| {
            let pw = submap.pose_world.transform_point(&p.cast::<f64>());
            pw.cast::<f32>()
        })
        .collect()
}