#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform TransformParams {
    float r00, r01, r02, t0;
    float r10, r11, r12, t1;
    float r20, r21, r22, t2;
    uint num_points;
} params;

layout(set = 0, binding = 0) readonly buffer InPoints {
    float pts[];
};
layout(set = 0, binding = 1) readonly buffer InCovs {
    float covs[];
};
layout(set = 0, binding = 2) writeonly buffer OutPoints {
    float out_pts[];
};
layout(set = 0, binding = 3) writeonly buffer OutCovs {
    float out_covs[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.num_points) return;

    uint pt_offset = idx * 3;
    float px = pts[pt_offset + 0];
    float py = pts[pt_offset + 1];
    float pz = pts[pt_offset + 2];

    out_pts[pt_offset + 0] = params.r00 * px + params.r01 * py + params.r02 * pz + params.t0;
    out_pts[pt_offset + 1] = params.r10 * px + params.r11 * py + params.r12 * pz + params.t1;
    out_pts[pt_offset + 2] = params.r20 * px + params.r21 * py + params.r22 * pz + params.t2;

    uint cov_offset = idx * 9;
    
    float c00 = covs[cov_offset + 0];
    float c01 = covs[cov_offset + 1];
    float c02 = covs[cov_offset + 2];
    float c11 = covs[cov_offset + 4];
    float c12 = covs[cov_offset + 5];
    float c22 = covs[cov_offset + 8];

    float tmp00 = params.r00 * c00 + params.r01 * c01 + params.r02 * c02;
    float tmp01 = params.r00 * c01 + params.r01 * c11 + params.r02 * c12;
    float tmp02 = params.r00 * c02 + params.r01 * c12 + params.r02 * c22;

    float tmp10 = params.r10 * c00 + params.r11 * c01 + params.r12 * c02;
    float tmp11 = params.r10 * c01 + params.r11 * c11 + params.r12 * c12;
    float tmp12 = params.r10 * c02 + params.r11 * c12 + params.r12 * c22;

    float tmp20 = params.r20 * c00 + params.r21 * c01 + params.r22 * c02;
    float tmp21 = params.r20 * c01 + params.r21 * c11 + params.r22 * c12;
    float tmp22 = params.r20 * c02 + params.r21 * c12 + params.r22 * c22;

    float tc00 = tmp00 * params.r00 + tmp01 * params.r01 + tmp02 * params.r02;
    float tc01 = tmp00 * params.r10 + tmp01 * params.r11 + tmp02 * params.r12;
    float tc02 = tmp00 * params.r20 + tmp01 * params.r21 + tmp02 * params.r22;
    
    float tc11 = tmp10 * params.r10 + tmp11 * params.r11 + tmp12 * params.r12;
    float tc12 = tmp10 * params.r20 + tmp11 * params.r21 + tmp12 * params.r22;
    
    float tc22 = tmp20 * params.r20 + tmp21 * params.r21 + tmp22 * params.r22;

    out_covs[cov_offset + 0] = tc00;
    out_covs[cov_offset + 1] = tc01;
    out_covs[cov_offset + 2] = tc02;

    out_covs[cov_offset + 3] = tc01; // tc10 = tc01
    out_covs[cov_offset + 4] = tc11;
    out_covs[cov_offset + 5] = tc12;

    out_covs[cov_offset + 6] = tc02; // tc20 = tc02
    out_covs[cov_offset + 7] = tc12; // tc21 = tc12
    out_covs[cov_offset + 8] = tc22;
}