#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint EMPTY_KEY = 0xFFFFFFFF;
const float FIXED_SCALE = 10000.0;  // Match the scale used in insert shader
const float INV_FIXED_SCALE = 1.0 / FIXED_SCALE;

layout(push_constant) uniform PushConstants {
    int num_points;
    int table_size;
    float voxel_size;
    int _pad;
} pc;

layout(set = 0, binding = 0) buffer InputPoints { float in_points[]; };
layout(set = 0, binding = 1) buffer TableKeys   { uint table_keys[]; };
layout(set = 0, binding = 2) buffer TableCentroids { uint table_centroids[]; };
layout(set = 0, binding = 3) buffer TableCounts { uint table_counts[]; };
layout(set = 0, binding = 4) buffer OutputPoints { float out_points[]; };
layout(set = 0, binding = 5) buffer OutputCount  { uint out_count; };
layout(set = 0, binding = 6) buffer TableVoxelIndices { uint table_voxel_indices[]; };
layout(set = 0, binding = 7) buffer ClusterIDs { uint cluster_ids[]; };


shared uint s_group_valid_count;
shared uint s_group_base_idx;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint l_idx = gl_LocalInvocationID.x;

    if (l_idx == 0) {
        s_group_valid_count = 0;
    }
    barrier();

    bool is_valid = false;
    uint count = 0;
    uint local_offset = 0;

    if (idx < pc.table_size) {
        uint key = table_keys[idx];
        count = table_counts[idx];

        if (key != EMPTY_KEY && count > 0) {
            is_valid = true;
            local_offset = atomicAdd(s_group_valid_count, 1);
        }
    }
    barrier();

    if (l_idx == 0) {
        if (s_group_valid_count > 0) {
            s_group_base_idx = atomicAdd(out_count, s_group_valid_count);
        }
    }
    barrier();

    if (is_valid) {
        int raw_x = int(table_centroids[idx * 3 + 0]);
        int raw_y = int(table_centroids[idx * 3 + 1]);
        int raw_z = int(table_centroids[idx * 3 + 2]);

        float sx = float(raw_x) * INV_FIXED_SCALE;
        float sy = float(raw_y) * INV_FIXED_SCALE;
        float sz = float(raw_z) * INV_FIXED_SCALE;

        if (count > 1) {
            float inv = 1.0 / float(count);
            sx *= inv;
            sy *= inv;
            sz *= inv;
        }

        uint w_idx = s_group_base_idx + local_offset;

        out_points[w_idx * 3 + 0] = sx;
        out_points[w_idx * 3 + 1] = sy;
        out_points[w_idx * 3 + 2] = sz;

        table_voxel_indices[idx] = w_idx;
        cluster_ids[w_idx] = w_idx;
    }
}
    