#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#define K 25
const float FLT_MAX = 3.402823466e+38;
const uint EMPTY_KEY = 0xFFFFFFFF;
const uint GLOBAL_PROBE = 1000;

const int GRID_OFFSET_X = 512;
const int GRID_OFFSET_Y = 512;
const int GRID_OFFSET_Z = 512;

const float FIXED_SCALE = 10000.0;  // Match the scale used in insert shader
const float INV_FIXED_SCALE = 1.0 / FIXED_SCALE;

layout(set = 0, binding = 0) restrict readonly buffer PointsBuffer {
    float points[];
};

layout(set = 0, binding = 1) restrict readonly buffer TableKeys {
    uint table_keys[];
};
layout(set = 0, binding = 2) restrict readonly buffer TableCentroids {
    uint table_centroids[];
};

layout(set = 0, binding = 3) restrict readonly buffer TableCounts {
    uint table_counts[]; 
};

layout(set = 0, binding = 4) restrict writeonly buffer CovarianceBuffer {
    float out_covariances[];
};

layout(push_constant) uniform PushConstants {
    int num_points;
    int table_size;
    float voxel_size;
    int _pad;
} pc;


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

int find_key_index(uint key) {
    uint h = (key * 2654435761u);
    uint idx = h % uint(pc.table_size);

    for (uint i = 0; i < GLOBAL_PROBE; ++i) {
        uint k = table_keys[idx];
        if (k == key) {
            return int(idx);
        } else if (k == EMPTY_KEY) {
            return -1;
        }
        idx = (idx + 1) % uint(pc.table_size);
    }
    return -1;
}

void eigen_decomposition_3x3(inout float A[3][3], out float evecs[3][3], out float evals[3]) {
    evecs[0][0] = 1.0; evecs[0][1] = 0.0; evecs[0][2] = 0.0;
    evecs[1][0] = 0.0; evecs[1][1] = 1.0; evecs[1][2] = 0.0;
    evecs[2][0] = 0.0; evecs[2][1] = 0.0; evecs[2][2] = 1.0;

    const int max_iter = 15;

    for (int iter = 0; iter < max_iter; ++iter) {
        int p = 0, q = 1;

        float a01 = abs(A[0][1]);
        float a02 = abs(A[0][2]);
        float a12 = abs(A[1][2]);

        float max_val;
        if (a01 >= a02 && a01 >= a12) {
            p = 0; q = 1; max_val = a01;
        } else if (a02 >= a01 && a02 >= a12) {
            p = 0; q = 2; max_val = a02;
        } else {
            p = 1; q = 2; max_val = a12;
        }

        if (max_val < 1e-6) break;

        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];

        float phi = 0.5 * atan(2.0 * apq, aqq - app);
        float c = cos(phi);
        float s = sin(phi);

        A[p][p] = c*c*app - 2.0*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2.0*s*c*apq + c*c*aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                float arp = A[r][p];
                float arq = A[r][q];
                A[r][p] = c*arp - s*arq;
                A[p][r] = A[r][p];
                A[r][q] = s*arp + c*arq;
                A[q][r] = A[r][q];
            }
        }

        for (int r = 0; r < 3; ++r) {
            float ervp = evecs[r][p];
            float ervq = evecs[r][q];
            evecs[r][p] = c*ervp - s*ervq;
            evecs[r][q] = s*ervp + c*ervq;
        }
    }

    evals[0] = A[0][0];
    evals[1] = A[1][1];
    evals[2] = A[2][2];
}

void recompute_max_k(in float dists[K], out float dmax, out int imax) {
    float dm = dists[0];
    int im = 0;
    for (int t = 1; t < K; ++t) {
        float v = dists[t];
        if (v > dm) { dm = v; im = t; }
    }
    dmax = dm;
    imax = im;
}

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.num_points) return;

    float px = points[idx * 3 + 0];
    float py = points[idx * 3 + 1];
    float pz = points[idx * 3 + 2];

    float neighbor_points[K * 3];
    float neighbor_dists[K];

    for (int i = 0; i < K; ++i) {
        neighbor_dists[i] = FLT_MAX;
    }

    float dmax = FLT_MAX;
    int imax = 0;

    float inv_voxel = 1.0 / pc.voxel_size;
    int cx = int(floor(px * inv_voxel)) + GRID_OFFSET_X;
    int cy = int(floor(py * inv_voxel)) + GRID_OFFSET_Y;
    int cz = int(floor(pz * inv_voxel)) + GRID_OFFSET_Z;

    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                
                int nx = clamp(cx + x, 0, 1023);
                int ny = clamp(cy + y, 0, 1023);
                int nz = clamp(cz + z, 0, 1023);

                uint key = morton3D(uvec3(nx, ny, nz));
                int table_idx = find_key_index(key);

                if (table_idx != -1) {
                    uint count = table_counts[table_idx];
                    
                    if (count > 0) {
                        int raw_sum_x = int(table_centroids[3 * table_idx + 0]);
                        int raw_sum_y = int(table_centroids[3 * table_idx + 1]);
                        int raw_sum_z = int(table_centroids[3 * table_idx + 2]);
                        
                        float div = INV_FIXED_SCALE / float(count);
                        float tx = float(raw_sum_x) * div;
                        float ty = float(raw_sum_y) * div;
                        float tz = float(raw_sum_z) * div;

                        float dx = px - tx;
                        float dy = py - ty;
                        float dz = pz - tz;
                        float d2 = dx*dx + dy*dy + dz*dz;

                        if (d2 < dmax) {
                            neighbor_dists[imax] = d2;
                            neighbor_points[imax * 3 + 0] = tx;
                            neighbor_points[imax * 3 + 1] = ty;
                            neighbor_points[imax * 3 + 2] = tz;
                            
                            recompute_max_k(neighbor_dists, dmax, imax);
                        }
                    }
                }
            }
        }
    }

    float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int valid_count = 0;

    for (int i = 0; i < K; ++i) {
        if (neighbor_dists[i] == FLT_MAX) continue;
        sum_x += neighbor_points[i * 3 + 0];
        sum_y += neighbor_points[i * 3 + 1];
        sum_z += neighbor_points[i * 3 + 2];
        valid_count++;
    }
    
    int base = idx * 9;
    if (valid_count < 3) {        
        out_covariances[base + 0] = 1.0; out_covariances[base + 1] = 0.0; out_covariances[base + 2] = 0.0;
        out_covariances[base + 3] = 0.0; out_covariances[base + 4] = 1.0; out_covariances[base + 5] = 0.0;
        out_covariances[base + 6] = 0.0; out_covariances[base + 7] = 0.0; out_covariances[base + 8] = 1.0;
        return;
    }

    float inv_n = 1.0 / float(valid_count);
    float mean_x = sum_x * inv_n;
    float mean_y = sum_y * inv_n;
    float mean_z = sum_z * inv_n;

    float c_xx = 0.0, c_xy = 0.0, c_xz = 0.0;
    float c_yy = 0.0, c_yz = 0.0, c_zz = 0.0;

    for (int i = 0; i < K; ++i) {
        if (neighbor_dists[i] == FLT_MAX) continue;

        float dx = neighbor_points[i * 3 + 0] - mean_x;
        float dy = neighbor_points[i * 3 + 1] - mean_y;
        float dz = neighbor_points[i * 3 + 2] - mean_z;

        c_xx += dx * dx; c_xy += dx * dy; c_xz += dx * dz;
        c_yy += dy * dy; c_yz += dy * dz; c_zz += dz * dz;
    }

    float mat[3][3];
    mat[0][0] = c_xx * inv_n; mat[0][1] = c_xy * inv_n; mat[0][2] = c_xz * inv_n;
    mat[1][0] = c_xy * inv_n; mat[1][1] = c_yy * inv_n; mat[1][2] = c_yz * inv_n;
    mat[2][0] = c_xz * inv_n; mat[2][1] = c_yz * inv_n; mat[2][2] = c_zz * inv_n;

    float evecs[3][3];
    float evals[3];
    eigen_decomposition_3x3(mat, evecs, evals);

    int min_idx = 0;
    if (evals[1] < evals[min_idx]) min_idx = 1;
    if (evals[2] < evals[min_idx]) min_idx = 2;

    // float reg_evals[3] = float[](evals[0], evals[1], evals[2]);
    float max_eval = max(evals[0], max(evals[1], evals[2]));
    float reg_evals[3];
    if (max_eval > 0.0) {
        reg_evals[0] = max(evals[0] / max_eval, 1e-6);
        reg_evals[1] = max(evals[1] / max_eval, 1e-6);
        reg_evals[2] = max(evals[2] / max_eval, 1e-6);
    } else {
        reg_evals[0] = 1.0; reg_evals[1] = 1.0; reg_evals[2] = 1.0;
    }

    // C = V * diag(reg_evals) * V^T
    float r00 = 0.0, r01 = 0.0, r02 = 0.0;
    float r11 = 0.0, r12 = 0.0;
    float r22 = 0.0;

    for (int k = 0; k < 3; ++k) {
        float lambda = reg_evals[k];
        float vx = evecs[0][k];
        float vy = evecs[1][k];
        float vz = evecs[2][k];

        r00 += lambda * vx * vx;
        r01 += lambda * vx * vy;
        r02 += lambda * vx * vz;
        r11 += lambda * vy * vy;
        r12 += lambda * vy * vz;
        r22 += lambda * vz * vz;
    }

    out_covariances[base + 0] = r00;
    out_covariances[base + 1] = r01;
    out_covariances[base + 2] = r02;

    out_covariances[base + 3] = r01;
    out_covariances[base + 4] = r11;
    out_covariances[base + 5] = r12;

    out_covariances[base + 6] = r02;
    out_covariances[base + 7] = r12;
    out_covariances[base + 8] = r22;
}

// 法線方向（最小固有値）を 1e-3 に正則化、残りは 1.0
    int min_idx = 0;
    if (evals[1] < evals[min_idx]) min_idx = 1;
    if (evals[2] < evals[min_idx]) min_idx = 2;

    float reg_evals[3];
    reg_evals[0] = 1.0;
    reg_evals[1] = 1.0;
    reg_evals[2] = 1.0;
    reg_evals[min_idx] = 1e-3;