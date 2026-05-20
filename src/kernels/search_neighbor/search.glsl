#version 450

const uint WORKGROUP_SIZE = 256;
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SearchParams {
    uint num_source;
    uint num_target;
} params;

layout(set = 0, binding = 0) readonly buffer SourcePts {
    float source_pts[];
};

layout(set = 0, binding = 1) readonly buffer TargetPts {
    float target_pts[];
};

layout(set = 0, binding = 2) writeonly buffer OutIndices {
    int out_indices[];
};

layout(set = 0, binding = 3) writeonly buffer OutDistsSq {
    float out_dists_sq[];
};

shared float s_target_x[WORKGROUP_SIZE];
shared float s_target_y[WORKGROUP_SIZE];
shared float s_target_z[WORKGROUP_SIZE];

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    float px = 0.0, py = 0.0, pz = 0.0;
    bool is_valid = idx < params.num_source;

    if (is_valid) {
        uint src_offset = idx * 3;
        px = source_pts[src_offset + 0];
        py = source_pts[src_offset + 1];
        pz = source_pts[src_offset + 2];
    };

    float best_dist_sq = 1.0e30;
    int best_target_idx = -1;

    uint num_chunks = (params.num_target + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        uint target_base_idx = chunk * WORKGROUP_SIZE;
        uint current_target_idx = target_base_idx + lid;

        if (current_target_idx < params.num_target) {
            uint tgt_offset = current_target_idx * 3;
            s_target_x[lid] = target_pts[tgt_offset + 0];
            s_target_y[lid] = target_pts[tgt_offset + 1];
            s_target_z[lid] = target_pts[tgt_offset + 2];
        }
        
        barrier();

        if (is_valid) {
            uint points_in_chunk = min(WORKGROUP_SIZE, params.num_target - target_base_idx);
            
            for (uint j = 0; j < points_in_chunk; j++) {
                float dx = px - s_target_x[j];
                float dy = py - s_target_y[j];
                float dz = pz - s_target_z[j];

                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_target_idx = int(target_base_idx + j);
                }
            }
        }
        
        barrier();
    }

    if (is_valid) {
        out_dists_sq[idx] = best_dist_sq;
        out_indices[idx] = best_target_idx;
    }
}