#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint EMPTY_KEY = 0xFFFFFFFF;
const uint SHARED_TABLE_SIZE = 1536;
const uint SHARED_PROBE = 64;
const uint GLOBAL_PROBE = 1000;

const float FIXED_SCALE = 10000.0;  // Match the scale used in compact shader
const float INV_FIXED_SCALE = 1.0 / FIXED_SCALE;

const int GRID_OFFSET_X = 512;
const int GRID_OFFSET_Y = 512;
const int GRID_OFFSET_Z = 512;

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


shared uint s_keys[SHARED_TABLE_SIZE];
shared int  s_centroids_x[SHARED_TABLE_SIZE];
shared int  s_centroids_y[SHARED_TABLE_SIZE];
shared int  s_centroids_z[SHARED_TABLE_SIZE];
shared int s_counts[SHARED_TABLE_SIZE];

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

void atomicAddGlobalFloat(uint index, float val) {
    uint assumed;
    uint old_val = table_centroids[index];
    do {
        assumed = old_val;
        float new_float = uintBitsToFloat(assumed) + val;
        uint new_val = floatBitsToUint(new_float);
        old_val = atomicCompSwap(table_centroids[index], assumed, new_val);
    } while (assumed != old_val);
}

uint compute_voxel_key(float px, float py, float pz, float voxel_size) {
    float inv_voxel = 1.0 / voxel_size;
    int vx = int(floor(px * inv_voxel)) + GRID_OFFSET_X;
    int vy = int(floor(py * inv_voxel)) + GRID_OFFSET_Y;
    int vz = int(floor(pz * inv_voxel)) + GRID_OFFSET_Z;

    vx = clamp(vx, 0, 1023);
    vy = clamp(vy, 0, 1023);
    vz = clamp(vz, 0, 1023);

    // return uint(vx) | (uint(vy) << 10) | (uint(vz) << 20);
    return morton3D(uvec3(vx, vy, vz));
}

void add_to_global_accumulated(uint key, int ix, int iy, int iz, int count) {
    uint h = (key * 2654435761u);
    uint idx = h % uint(pc.table_size);

    for (uint i = 0; i < GLOBAL_PROBE; ++i) {
        uint current_key = table_keys[idx];

        if (current_key == key) {
            atomicAdd(table_centroids[3 * idx + 0], uint(ix));
            atomicAdd(table_centroids[3 * idx + 1], uint(iy));
            atomicAdd(table_centroids[3 * idx + 2], uint(iz));
            atomicAdd(table_counts[idx], count);
            return;
        } else if (current_key == EMPTY_KEY) {
            uint old = atomicCompSwap(table_keys[idx], EMPTY_KEY, key);
            if (old == EMPTY_KEY || old == key) {
                atomicAdd(table_centroids[3 * idx + 0], uint(ix));
                atomicAdd(table_centroids[3 * idx + 1], uint(iy));
                atomicAdd(table_centroids[3 * idx + 2], uint(iz));
                atomicAdd(table_counts[idx], count);
                return;
            }
        }
        idx = (idx + 1) % uint(pc.table_size);
    }
}

void main() {
    uint g_idx = gl_GlobalInvocationID.x;
    uint l_idx = gl_LocalInvocationID.x;
    uint l_size = gl_WorkGroupSize.x;

    for (uint i = l_idx; i < SHARED_TABLE_SIZE; i += l_size) {
        s_keys[i] = EMPTY_KEY;
        s_centroids_x[i] = 0;
        s_centroids_y[i] = 0;
        s_centroids_z[i] = 0;
        s_counts[i] = 0;
    }
    barrier();

    if (g_idx < pc.num_points) {
        float px = in_points[g_idx * 3 + 0];
        float py = in_points[g_idx * 3 + 1];
        float pz = in_points[g_idx * 3 + 2];
        uint key = compute_voxel_key(px, py, pz, pc.voxel_size);

        uint s_idx = key % SHARED_TABLE_SIZE;
        bool stored = false;

        int ix = int(px * FIXED_SCALE);
        int iy = int(py * FIXED_SCALE);
        int iz = int(pz * FIXED_SCALE);

        for (uint i = 0; i < SHARED_PROBE; ++i) {
            uint old = atomicCompSwap(s_keys[s_idx], EMPTY_KEY, key);
            
            if (old == EMPTY_KEY || old == key) {
                atomicAdd(s_centroids_x[s_idx], ix);
                atomicAdd(s_centroids_y[s_idx], iy);
                atomicAdd(s_centroids_z[s_idx], iz);
                atomicAdd(s_counts[s_idx], 1);
                stored = true;
                break;
            }
            s_idx = (s_idx + 1) % SHARED_TABLE_SIZE;
        }

        if (!stored) {
            add_to_global_accumulated(key, ix, iy, iz, 1);
        }
    }
    barrier();

    for (uint i = l_idx; i < SHARED_TABLE_SIZE; i += l_size) {
        uint key = s_keys[i];
        if (key != EMPTY_KEY) {
            int sx = s_centroids_x[i];
            int sy = s_centroids_y[i];
            int sz = s_centroids_z[i];
            int sc = s_counts[i];
            add_to_global_accumulated(key, sx, sy, sz, sc);
        }
    }
}