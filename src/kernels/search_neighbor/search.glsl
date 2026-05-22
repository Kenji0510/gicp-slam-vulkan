#version 450

const uint WORKGROUP_SIZE = 256;
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

const uint EMPTY_KEY = 0xFFFFFFFFu;
const uint GLOBAL_PROBE = 1000u;

const int GRID_OFFSET_X = 512;
const int GRID_OFFSET_Y = 512;
const int GRID_OFFSET_Z = 512;

layout(push_constant) uniform SearchParams {
    uint num_source;
    uint num_target;
    uint table_size;
    int search_range;
    float voxel_size;
    float max_dist_sq;
} params;

// transformed source points
layout(set = 0, binding = 0) readonly buffer SourcePts {
    float source_pts[];
};

// compacted target points
layout(set = 0, binding = 1) readonly buffer TargetPts {
    float target_pts[];
};

// target voxel hash table keys
layout(set = 0, binding = 2) readonly buffer TargetTableKeys {
    uint target_table_keys[];
};

// hash table slot -> compacted target point index
layout(set = 0, binding = 3) readonly buffer TargetTableVoxelIndices {
    uint target_table_voxel_indices[];
};

layout(set = 0, binding = 4) writeonly buffer OutIndices {
    int out_indices[];
};

layout(set = 0, binding = 5) writeonly buffer OutDistsSq {
    float out_dists_sq[];
};

uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint morton3D(uvec3 v) {
    return expandBits(v.x) | (expandBits(v.y) << 1) | (expandBits(v.z) << 2);
}

uint voxel_key_from_index(int vx, int vy, int vz) {
    int ix = clamp(vx + GRID_OFFSET_X, 0, 1023);
    int iy = clamp(vy + GRID_OFFSET_Y, 0, 1023);
    int iz = clamp(vz + GRID_OFFSET_Z, 0, 1023);

    return morton3D(uvec3(uint(ix), uint(iy), uint(iz)));
}

int find_key_index(uint key) {
    uint h = key * 2654435761u;
    uint idx = h % params.table_size;

    for (uint i = 0; i < GLOBAL_PROBE; ++i) {
        uint k = target_table_keys[idx];

        if (k == key) {
            return int(idx);
        }

        if (k == EMPTY_KEY) {
            return -1;
        }

        idx = (idx + 1u) % params.table_size;
    }

    return -1;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= params.num_source) {
        return;
    }

    uint src_offset = idx * 3u;

    float px = source_pts[src_offset + 0u];
    float py = source_pts[src_offset + 1u];
    float pz = source_pts[src_offset + 2u];

    float inv_voxel = 1.0 / params.voxel_size;

    int base_vx = int(floor(px * inv_voxel));
    int base_vy = int(floor(py * inv_voxel));
    int base_vz = int(floor(pz * inv_voxel));

    float best_dist_sq = 1.0e30;
    int best_target_idx = -1;

    for (int dz = -params.search_range; dz <= params.search_range; ++dz) {
        for (int dy = -params.search_range; dy <= params.search_range; ++dy) {
            for (int dx = -params.search_range; dx <= params.search_range; ++dx) {
                int vx = base_vx + dx;
                int vy = base_vy + dy;
                int vz = base_vz + dz;

                uint key = voxel_key_from_index(vx, vy, vz);
                int table_idx = find_key_index(key);

                if (table_idx < 0) {
                    continue;
                }

                uint target_idx_u = target_table_voxel_indices[table_idx];

                if (target_idx_u == EMPTY_KEY || target_idx_u >= params.num_target) {
                    continue;
                }

                uint tgt_offset = target_idx_u * 3u;

                float tx = target_pts[tgt_offset + 0u];
                float ty = target_pts[tgt_offset + 1u];
                float tz = target_pts[tgt_offset + 2u];

                float ex = px - tx;
                float ey = py - ty;
                float ez = pz - tz;

                float dist_sq = ex * ex + ey * ey + ez * ez;

                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_target_idx = int(target_idx_u);
                }
            }
        }
    }

    if (params.max_dist_sq > 0.0 && best_dist_sq > params.max_dist_sq) {
        out_indices[idx] = -1;
        out_dists_sq[idx] = best_dist_sq;
        return;
    }

    out_indices[idx] = best_target_idx;
    out_dists_sq[idx] = best_dist_sq;
}