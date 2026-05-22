use std::path::PathBuf;

use plotters::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceLogs {
    pub voxelization_time_ms: Vec<f32>,
    pub create_voxel_map_time_ms: Vec<f32>,
    pub knn_search_time_ms: Vec<f32>,
    pub compute_covariances_time_ms: Vec<f32>,
    pub transform_points_time_ms: Vec<f32>,
    pub find_neighbors_time_ms: Vec<f32>,
    pub each_gicp_time_ms: Vec<f32>,
    pub total_gicp_time_ms: Vec<f32>,
    pub query_voxel_time_ms: Vec<f32>,
    pub update_voxel_map_time_ms: Vec<f32>,
    pub merge_time_ms: Vec<f32>,
    pub total_average_time_ms: Vec<f32>,
    pub iteration_count: usize,
}

/// `each_gicp_time_ms` は GICP_ITERATIONS 本まとめて記録されているため、
/// フレームごとの合計に畳み込む。
fn fold_gicp(v: &[f32], gicp_iters: usize) -> Vec<f32> {
    if gicp_iters == 0 {
        return v.to_vec();
    }
    v.chunks(gicp_iters).map(|c| c.iter().sum()).collect()
}

struct Series<'a> {
    label: &'a str,
    data: Vec<f32>,
    color: RGBColor,
}

fn plot_series(series: &[Series], out_path: &str, title: &str) -> anyhow::Result<()> {
    let max_len = series.iter().map(|s| s.data.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(());
    }

    let y_max = series
        .iter()
        .flat_map(|s| s.data.iter().copied())
        .fold(0.0_f32, f32::max)
        * 1.1;

    let root = BitMapBackend::new(out_path, (1280, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..max_len, 0.0_f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Time [ms]")
        .draw()?;

    for s in series {
        let data: Vec<(usize, f32)> = s.data.iter().copied().enumerate().collect();
        chart
            .draw_series(LineSeries::new(data, s.color.stroke_width(2)))?
            .label(s.label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], s.color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Saved: {}", out_path);
    Ok(())
}

const VOXEL_SIZE: f32 = 0.5;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let default_path = format!(
        "data/output/05212026/debug/performance_logs_v-{}.json",
        VOXEL_SIZE
    );
    let json_path = args.get(1).map(|s| s.as_str()).unwrap_or(&default_path);

    let out_dir = PathBuf::from(json_path)
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .to_path_buf();

    let json = std::fs::read_to_string(json_path)?;
    let logs: PerformanceLogs = serde_json::from_str(&json)?;

    // GICP_ITERATIONS を each_gicp_time_ms の長さから推定
    let gicp_iters = if logs.iteration_count > 0 {
        logs.each_gicp_time_ms.len() / logs.iteration_count
    } else {
        1
    };
    let each_gicp_per_frame = fold_gicp(&logs.each_gicp_time_ms, gicp_iters);

    // --- グラフ 1: フレームごとの各処理時間 ---
    plot_series(
        &[
            Series {
                label: "Voxelization",
                data: logs.voxelization_time_ms.clone(),
                color: RED,
            },
            Series {
                label: "Query local map",
                data: logs.create_voxel_map_time_ms.clone(),
                color: BLUE,
            },
            Series {
                label: "KNN search",
                data: logs.knn_search_time_ms.clone(),
                color: GREEN,
            },
            Series {
                label: "Covariance",
                data: logs.compute_covariances_time_ms.clone(),
                color: CYAN,
            },
            Series {
                label: "Total GICP",
                data: logs.total_gicp_time_ms.clone(),
                color: MAGENTA,
            },
            Series {
                label: "Map update",
                data: logs.update_voxel_map_time_ms.clone(),
                color: RGBColor(255, 140, 0),
            },
        ],
        &out_dir
            .join(format!("perf_per_frame_v-{}.png", VOXEL_SIZE))
            .to_string_lossy(),
        "Per-frame processing time",
    )?;

    // --- グラフ 2: GICP 内訳 (transform / neighbor / gicp_solve) ---
    plot_series(
        &[
            Series {
                label: "Transform (per call)",
                data: logs.transform_points_time_ms.clone(),
                color: RED,
            },
            Series {
                label: "Neighbor search (per call)",
                data: logs.find_neighbors_time_ms.clone(),
                color: BLUE,
            },
            Series {
                label: "GICP solve (per call)",
                data: logs.each_gicp_time_ms.clone(),
                color: GREEN,
            },
            Series {
                label: "GICP total (per frame)",
                data: each_gicp_per_frame,
                color: MAGENTA,
            },
        ],
        &out_dir
            .join(format!("perf_gicp_detail_v-{}.png", VOXEL_SIZE))
            .to_string_lossy(),
        "GICP iteration detail",
    )?;

    Ok(())
}
