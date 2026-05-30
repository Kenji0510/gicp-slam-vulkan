use nalgebra::{Isometry3, Matrix6};
use rustc_hash::FxHashMap;

use crate::{
    loop_closure::LoopConstraintCandidate,
    submap::{Submap, SubmapId, SubmapManager},
};

#[derive(Debug, Clone)]
pub struct PoseGraphNode {
    pub submap_id: SubmapId,

    /// submap local -> world
    pub pose_world: Isometry3<f64>,

    /// 最初のsubmapなど、固定したいnode
    pub fixed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseGraphEdgeKind {
    Odometry,
    LoopClosure,
}

#[derive(Debug, Clone)]
pub struct PoseGraphEdge {
    pub from_id: SubmapId,
    pub to_id: SubmapId,

    /// from -> to の相対姿勢
    ///
    /// T_from_to = pose_from^-1 * pose_to
    pub relative_pose: Isometry3<f64>,

    pub information: Matrix6<f64>,

    pub kind: PoseGraphEdgeKind,
}

pub struct PoseGraph {
    pub nodes: FxHashMap<SubmapId, PoseGraphNode>,
    pub edges: Vec<PoseGraphEdge>,
}

impl PoseGraph {
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            edges: Vec::new(),
        }
    }

    pub fn add_node_from_submap(&mut self, submap: &Submap) {
        self.nodes.entry(submap.id).or_insert(PoseGraphNode {
            submap_id: submap.id,
            pose_world: submap.pose_world,
            fixed: submap.id == 0,
        });
    }

    pub fn add_odometry_edge(
        &mut self,
        prev: &Submap,
        curr: &Submap,
        information: Matrix6<f64>,
    ) {
        let relative_pose = prev.pose_world.inverse() * curr.pose_world;

        self.edges.push(PoseGraphEdge {
            from_id: prev.id,
            to_id: curr.id,
            relative_pose,
            information,
            kind: PoseGraphEdgeKind::Odometry,
        });
    }

    pub fn add_loop_constraint(
        &mut self,
        constraint: &LoopConstraintCandidate,
    ) {
        self.nodes.entry(constraint.candidate_id).or_insert(PoseGraphNode {
            submap_id: constraint.candidate_id,
            pose_world: Isometry3::identity(),
            fixed: constraint.candidate_id == 0,
        });

        self.nodes.entry(constraint.current_id).or_insert(PoseGraphNode {
            submap_id: constraint.current_id,
            pose_world: constraint.corrected_current_pose_world,
            fixed: constraint.current_id == 0,
        });

        self.edges.push(PoseGraphEdge {
            from_id: constraint.candidate_id,
            to_id: constraint.current_id,
            relative_pose: constraint.relative_pose_candidate_to_current,
            information: constraint.information,
            kind: PoseGraphEdgeKind::LoopClosure,
        });
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn loop_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.kind == PoseGraphEdgeKind::LoopClosure)
            .count()
    }

    pub fn odometry_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.kind == PoseGraphEdgeKind::Odometry)
            .count()
    }

    /// まだ本格的なgraph optimizationを入れない段階用。
    /// loop edgeが追加されたことだけを確認する。
    pub fn print_summary(&self) {
        log::info!(
            "PoseGraph: nodes={}, edges={}, odom_edges={}, loop_edges={}",
            self.node_count(),
            self.edge_count(),
            self.odometry_edge_count(),
            self.loop_edge_count(),
        );
    }

    /// 将来的に optimizer の結果を SubmapManager 側へ反映するための関数。
    /// まだ最適化しない段階では呼ばなくてOK。
    pub fn apply_poses_to_submaps(&self, submap_manager: &mut SubmapManager) {
        for submap in &mut submap_manager.submaps {
            if let Some(node) = self.nodes.get(&submap.id) {
                submap.pose_world = node.pose_world;
                submap.center_world = nalgebra::Point3::new(
                    node.pose_world.translation.vector.x as f32,
                    node.pose_world.translation.vector.y as f32,
                    node.pose_world.translation.vector.z as f32,
                );
            }
        }
    }
}

pub fn default_odometry_information() -> Matrix6<f64> {
    let mut info = Matrix6::<f64>::identity();

    // 回転・並進とも仮の重み。
    // 後でGICPのHessianや経験値から調整してよい。
    for i in 0..6 {
        info[(i, i)] = 1.0;
    }

    info
}

pub fn default_loop_information() -> Matrix6<f64> {
    let mut info = Matrix6::<f64>::identity();

    // loop edgeは強めにしたいが、最初は大きくしすぎない。
    for i in 0..6 {
        info[(i, i)] = 10.0;
    }

    info
}