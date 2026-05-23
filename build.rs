use std::{
    env,
    fs,
    path::{Path, PathBuf},
};

fn compile_shader(
    compiler: &shaderc::Compiler,
    options: &shaderc::CompileOptions,
    src_path: &Path,
    out_path: &Path,
) {
    let src = fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", src_path, e));

    let artifact = compiler
        .compile_into_spirv(
            &src,
            shaderc::ShaderKind::Compute,
            src_path.to_str().unwrap(),
            "main",
            Some(options),
        )
        .unwrap_or_else(|e| panic!("Failed to compile shader {:?}: {}", src_path, e));

    fs::write(out_path, artifact.as_binary_u8())
        .unwrap_or_else(|e| panic!("Failed to write SPIR-V to {:?}: {}", out_path, e));

    // Trigger recompilation when the shader source changes
    println!("cargo:rerun-if-changed={}", src_path.display());
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_out_dir = out_dir.join("shaders");
    fs::create_dir_all(&shader_out_dir).unwrap();

    let compiler = shaderc::Compiler::new().expect("Failed to create shaderc compiler");

    let mut options = shaderc::CompileOptions::new().expect("Failed to create compile options");
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );

    let shaders: &[(&str, &str)] = &[
        ("src/kernels/gicp/gicp.glsl",                     "gicp.spv"),
        ("src/kernels/search_neighbor/search.glsl",        "search_neighbor.spv"),
        ("src/kernels/voxelization/init.glsl",             "voxel_init.spv"),
        ("src/kernels/voxelization/insert.glsl",           "voxel_insert.spv"),
        ("src/kernels/voxelization/compact.glsl",          "voxel_compact.spv"),
        ("src/kernels/covariances/covariance.glsl",        "covariance.spv"),
        ("src/kernels/knn_search/knn_search.glsl",         "knn_search.spv"),
        ("src/kernels/transform/transform.glsl",           "transform.spv"),
    ];

    for (src_rel, out_name) in shaders {
        let src_path = Path::new(src_rel);
        let out_path = shader_out_dir.join(out_name);
        compile_shader(&compiler, &options, src_path, &out_path);
    }
}
