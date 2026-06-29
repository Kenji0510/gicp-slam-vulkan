use std::path::PathBuf;

use plotters::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoopClosureRmseEntry {
    pub frame: usize,
    pub current_submap_id: u64,
    pub candidate_submap_id: u64,
    pub rmse: f32,
    pub valid_ratio: f32,
}

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
    #[serde(default)]
    pub gicp_rmse_per_frame: Vec<Option<f32>>,
    #[serde(default)]
    pub loop_closure_rmse: Vec<LoopClosureRmseEntry>,
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

/// `gicp_rmse_per_frame` (Vec<Option<f32>>) を折れ線グラフとして描画する。
/// None のフレームはスキップ（線が切れる）。
fn plot_gicp_rmse(rmse: &[Option<f32>], out_path: &str) -> anyhow::Result<()> {
    let valid: Vec<(usize, f32)> = rmse
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|r| (i, r)))
        .collect();

    if valid.is_empty() {
        return Ok(());
    }

    let y_max = valid.iter().map(|&(_, v)| v).fold(0.0_f32, f32::max) * 1.2;
    let y_max = if y_max < 1e-6 { 1.0 } else { y_max };

    let root = BitMapBackend::new(out_path, (1280, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("GICP final RMSE per frame", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..rmse.len(), 0.0_f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("RMSE [m]")
        .draw()?;

    chart
        .draw_series(LineSeries::new(valid.clone(), RED.stroke_width(2)))?
        .label("GICP RMSE")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    // スキップフレームを × マーカーで示す
    let skipped: Vec<(usize, f32)> = rmse
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if v.is_none() { Some((i, 0.0)) } else { None })
        .collect();
    if !skipped.is_empty() {
        chart
            .draw_series(skipped.iter().map(|&(x, y)| {
                Cross::new((x, y), 6, BLUE.filled())
            }))?
            .label("Skipped frame")
            .legend(|(x, y)| Cross::new((x, y), 6, BLUE.filled()));
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

/// loop closure RMSE をフレームインデックス × RMSE の散布図として描画する。
fn plot_loop_closure_rmse(entries: &[LoopClosureRmseEntry], total_frames: usize, out_path: &str) -> anyhow::Result<()> {
    if entries.is_empty() {
        println!("No loop closure events — skipping {}", out_path);
        return Ok(());
    }

    let x_max = total_frames.max(entries.iter().map(|e| e.frame).max().unwrap_or(0) + 1);

    // フレームインデックス → RMSE / valid_ratio のマップを作成
    let rmse_at: std::collections::HashMap<usize, f32> =
        entries.iter().map(|e| (e.frame, e.rmse)).collect();
    let ratio_at: std::collections::HashMap<usize, f32> =
        entries.iter().map(|e| (e.frame, e.valid_ratio)).collect();

    // loop closure が発生したフレームのみ連結した折れ線データを生成
    let rmse_line: Vec<(usize, f32)> = {
        let mut v: Vec<(usize, f32)> = entries.iter().map(|e| (e.frame, e.rmse)).collect();
        v.sort_by_key(|&(f, _)| f);
        v
    };
    let ratio_line: Vec<(usize, f32)> = {
        let mut v: Vec<(usize, f32)> = entries.iter().map(|e| (e.frame, e.valid_ratio)).collect();
        v.sort_by_key(|&(f, _)| f);
        v
    };

    let y_max_rmse = rmse_line.iter().map(|&(_, v)| v).fold(0.0_f32, f32::max) * 1.2;
    let y_max_rmse = if y_max_rmse < 1e-6 { 1.0 } else { y_max_rmse };

    // --- RMSE 折れ線 ---
    {
        let root = BitMapBackend::new(out_path, (1280, 640)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Loop closure RMSE per event", ("sans-serif", 22))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(0usize..x_max, 0.0_f32..y_max_rmse)?;

        chart
            .configure_mesh()
            .x_desc("Frame")
            .y_desc("RMSE [m]")
            .draw()?;

        // 折れ線
        chart
            .draw_series(LineSeries::new(rmse_line.clone(), MAGENTA.stroke_width(2)))?
            .label("Loop closure RMSE")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA.stroke_width(2)));

        // イベント点を丸マーカーで強調
        chart.draw_series(rmse_line.iter().map(|&(x, y)| {
            Circle::new((x, y), 5, MAGENTA.filled())
        }))?;

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        println!("Saved: {}", out_path);
    }

    // --- valid_ratio 折れ線（別ファイル） ---
    let ratio_path = out_path.replace(".png", "_valid_ratio.png");
    {
        let root = BitMapBackend::new(&ratio_path, (1280, 640)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Loop closure valid ratio per event", ("sans-serif", 22))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(0usize..x_max, 0.0_f32..1.05_f32)?;

        chart
            .configure_mesh()
            .x_desc("Frame")
            .y_desc("Valid ratio")
            .draw()?;

        chart
            .draw_series(LineSeries::new(ratio_line.clone(), BLUE.stroke_width(2)))?
            .label("Valid ratio")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

        chart.draw_series(ratio_line.iter().map(|&(x, y)| {
            Circle::new((x, y), 5, BLUE.filled())
        }))?;

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        println!("Saved: {}", ratio_path);
    }

    Ok(())
}

const VOXEL_SIZE: f32 = 0.2;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let default_path = format!(
        "data/output/06212026/debug/performance_logs_v-{}.json",
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

    // --- グラフ 3: odometry GICP RMSE per frame ---
    plot_gicp_rmse(
        &logs.gicp_rmse_per_frame,
        &out_dir
            .join(format!("gicp_rmse_per_frame_v-{}.png", VOXEL_SIZE))
            .to_string_lossy(),
    )?;

    // --- グラフ 4: loop closure RMSE ---
    plot_loop_closure_rmse(
        &logs.loop_closure_rmse,
        logs.iteration_count,
        &out_dir
            .join(format!("loop_closure_rmse_v-{}.png", VOXEL_SIZE))
            .to_string_lossy(),
    )?;

    Ok(())
}
