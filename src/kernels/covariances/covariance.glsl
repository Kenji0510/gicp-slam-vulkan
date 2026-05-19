#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint K = 15;

layout(push_constant) uniform Params {
    uint num_points;
} params;

layout(std430, set = 0, binding = 0) readonly buffer Points {
    float points[];
};

layout(std430, set = 0, binding = 1) readonly buffer NeighborIndices {
    int neighbor_indices[];
};

// 9 floats per point (row-major 3x3)
layout(std430, set = 0, binding = 2) writeonly buffer OutCovariances {
    float out_covariances[];
};

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

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.num_points) {
        return;
    }

    // 近傍点の収集と重心計算
    float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int valid_count = 0;

    float cache_x[K];
    float cache_y[K];
    float cache_z[K];

    uint neighbor_base = idx * K;

    for (uint i = 0; i < K; ++i) {
        int n_idx = neighbor_indices[neighbor_base + i];

        if (n_idx < 0 || n_idx >= int(params.num_points)) {
            break;
        }

        uint pt_offset = uint(n_idx) * 3;
        float tx = points[pt_offset + 0];
        float ty = points[pt_offset + 1];
        float tz = points[pt_offset + 2];

        cache_x[valid_count] = tx;
        cache_y[valid_count] = ty;
        cache_z[valid_count] = tz;

        sum_x += tx;
        sum_y += ty;
        sum_z += tz;
        valid_count++;
    }

    uint out_base = idx * 9;

    // 近傍点が足りない場合は単位行列を格納
    if (valid_count < 3) {
        out_covariances[out_base + 0] = 1.0; out_covariances[out_base + 1] = 0.0; out_covariances[out_base + 2] = 0.0;
        out_covariances[out_base + 3] = 0.0; out_covariances[out_base + 4] = 1.0; out_covariances[out_base + 5] = 0.0;
        out_covariances[out_base + 6] = 0.0; out_covariances[out_base + 7] = 0.0; out_covariances[out_base + 8] = 1.0;
        return;
    }

    float inv_n = 1.0 / float(valid_count);
    float mean_x = sum_x * inv_n;
    float mean_y = sum_y * inv_n;
    float mean_z = sum_z * inv_n;

    // 共分散行列の計算
    float mat[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat[i][j] = 0.0;

    for (int i = 0; i < valid_count; ++i) {
        float dx = cache_x[i] - mean_x;
        float dy = cache_y[i] - mean_y;
        float dz = cache_z[i] - mean_z;

        mat[0][0] += dx * dx;
        mat[0][1] += dx * dy;
        mat[0][2] += dx * dz;
        mat[1][1] += dy * dy;
        mat[1][2] += dy * dz;
        mat[2][2] += dz * dz;
    }

    mat[0][0] *= inv_n; mat[0][1] *= inv_n; mat[0][2] *= inv_n;
    mat[1][1] *= inv_n; mat[1][2] *= inv_n; mat[2][2] *= inv_n;

    mat[1][0] = mat[0][1];
    mat[2][0] = mat[0][2];
    mat[2][1] = mat[1][2];

    // 固有分解
    float evecs[3][3];
    float evals[3];
    eigen_decomposition_3x3(mat, evecs, evals);

    // 法線方向（最小固有値）を 1e-3 に正則化、残りは 1.0
    int min_idx = 0;
    if (evals[1] < evals[min_idx]) min_idx = 1;
    if (evals[2] < evals[min_idx]) min_idx = 2;

    float reg_evals[3];
    reg_evals[0] = 1.0;
    reg_evals[1] = 1.0;
    reg_evals[2] = 1.0;
    reg_evals[min_idx] = 1e-3;

    // C = V * diag(reg_evals) * V^T の再構築
    float r_cov[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r_cov[i][j] = 0.0;

    for (int k = 0; k < 3; ++k) {
        float lambda = reg_evals[k];
        float vx = evecs[0][k];
        float vy = evecs[1][k];
        float vz = evecs[2][k];

        r_cov[0][0] += lambda * vx * vx;
        r_cov[0][1] += lambda * vx * vy;
        r_cov[0][2] += lambda * vx * vz;
        r_cov[1][1] += lambda * vy * vy;
        r_cov[1][2] += lambda * vy * vz;
        r_cov[2][2] += lambda * vz * vz;
    }

    // 対称成分の補完
    r_cov[1][0] = r_cov[0][1];
    r_cov[2][0] = r_cov[0][2];
    r_cov[2][1] = r_cov[1][2];

    // グローバルメモリへ書き出し (row-major)
    out_covariances[out_base + 0] = r_cov[0][0];
    out_covariances[out_base + 1] = r_cov[0][1];
    out_covariances[out_base + 2] = r_cov[0][2];
    out_covariances[out_base + 3] = r_cov[1][0];
    out_covariances[out_base + 4] = r_cov[1][1];
    out_covariances[out_base + 5] = r_cov[1][2];
    out_covariances[out_base + 6] = r_cov[2][0];
    out_covariances[out_base + 7] = r_cov[2][1];
    out_covariances[out_base + 8] = r_cov[2][2];
}
