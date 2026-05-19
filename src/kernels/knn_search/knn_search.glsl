#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint K = 15;

layout(push_constant) uniform KnnParams {
    uint num_points;
} params;

// n * 3
layout(set = 0, binding = 0) readonly buffer Pts {
    float pts[];
};

// n * k
layout(set = 0, binding = 1) writeonly buffer OutIndices {
    int out_indices[];
};

// n * k
layout(set = 0, binding = 2) writeonly buffer OutDistsSq {
    float out_dists_sq[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    if (gid >= params.num_points) return;

    uint pt_offset = gid * 3;
    float px = pts[pt_offset + 0];
    float py = pts[pt_offset + 1];
    float pz = pts[pt_offset + 2];

    float best_dists[K];
    int best_indices[K];

    for (uint i = 0; i < K; i++) {
        best_dists[i] = 1.0e30;
        best_indices[i] = -1;
    }

    for (uint j = 0; j < params.num_points; j++) {
        uint target_offset = j * 3;
        float tx = pts[target_offset + 0];
        float ty = pts[target_offset + 1];
        float tz = pts[target_offset + 2];

        float dx = px - tx;
        float dy = py - ty;
        float dz = pz - tz;

        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < best_dists[K - 1]) {
            int insert_pos = int(K) - 1;

            while (insert_pos > 0 && dist_sq < best_dists[insert_pos - 1]) {
                best_dists[insert_pos] = best_dists[insert_pos - 1];
                best_indices[insert_pos] = best_indices[insert_pos - 1];
                insert_pos--;
            }

            best_dists[insert_pos] = dist_sq;
            best_indices[insert_pos] = int(j);
        }
    }

    uint base_idx = gid * K;
    for (uint i = 0; i < K; i++) {
        out_indices[base_idx + i] = best_indices[i];
        out_dists_sq[base_idx + i] = best_dists[i];
    }
}