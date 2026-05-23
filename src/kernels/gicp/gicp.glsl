#version 450

const uint BLOCK_SIZE = 64;
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform IcpParams {
    int num_source;
    int num_target;
    float max_dist_sq;
} params;

layout(set = 0, binding = 0) readonly buffer SourcePts  { float source_pts[];  };
layout(set = 0, binding = 1) readonly buffer SourceCovs { float source_covs[]; };
layout(set = 0, binding = 2) readonly buffer TargetPts  { float target_pts[];  };
layout(set = 0, binding = 3) readonly buffer TargetCovs { float target_covs[]; };
layout(set = 0, binding = 4) readonly buffer Indices    { int   indices[];      };
layout(set = 0, binding = 5) readonly buffer Distances  { float dists_sq[];     };
// ワークグループごとの部分和バッファ（グローバルアトミック不要）
// レイアウト: [wg_id * 36 + i] for H,  [wg_id * 6 + i] for b
layout(set = 0, binding = 6) buffer OutPartialH { float d_partial_H[]; };
layout(set = 0, binding = 7) buffer OutPartialB { float d_partial_b[]; };

// ---- shared memory (per-block accumulator) ----
shared uint s_H[36];
shared uint s_b[6];

void atomicAddSharedH(uint idx, float val) {
    uint old_val = s_H[idx];
    uint assumed;
    do {
        assumed = old_val;
        old_val = atomicCompSwap(s_H[idx], assumed,
                      floatBitsToUint(uintBitsToFloat(assumed) + val));
    } while (assumed != old_val);
}

void atomicAddSharedB(uint idx, float val) {
    uint old_val = s_b[idx];
    uint assumed;
    do {
        assumed = old_val;
        old_val = atomicCompSwap(s_b[idx], assumed,
                      floatBitsToUint(uintBitsToFloat(assumed) + val));
    } while (assumed != old_val);
}

// ---- helpers ----
bool invert3x3Sym(float src[9], out float dst[9]) {
    float det = src[0] * (src[4]*src[8] - src[5]*src[7])
              - src[1] * (src[3]*src[8] - src[5]*src[6])
              + src[2] * (src[3]*src[7] - src[4]*src[6]);
    if (abs(det) < 1e-12) return false;
    float inv = 1.0 / det;
    dst[0] = (src[4]*src[8] - src[5]*src[7]) * inv;
    dst[1] = (src[2]*src[7] - src[1]*src[8]) * inv;
    dst[2] = (src[1]*src[5] - src[2]*src[4]) * inv;
    dst[3] = dst[1];
    dst[4] = (src[0]*src[8] - src[2]*src[6]) * inv;
    dst[5] = (src[2]*src[3] - src[0]*src[5]) * inv;
    dst[6] = dst[2];
    dst[7] = dst[5];
    dst[8] = (src[0]*src[4] - src[1]*src[3]) * inv;
    return true;
}

void mul3(float A[9], float v[3], out float r[3]) {
    r[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];
    r[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];
    r[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    // shared memory の初期化
    if (lid < 36u) s_H[lid] = 0u;
    if (lid < 6u)  s_b[lid] = 0u;
    barrier();

    float local_H[36];
    float local_b[6];
    for (int i = 0; i < 36; i++) local_H[i] = 0.0;
    for (int i = 0; i < 6;  i++) local_b[i] = 0.0;

    if (gid < uint(params.num_source)) {
        int target_idx = indices[gid];
        float dist_sq  = dists_sq[gid];

        if (target_idx >= 0 && uint(target_idx) < uint(params.num_target) && dist_sq <= params.max_dist_sq) {
            uint src_off = gid * 3u;
            float ps_x = source_pts[src_off + 0u];
            float ps_y = source_pts[src_off + 1u];
            float ps_z = source_pts[src_off + 2u];

            uint tgt_off = uint(target_idx) * 3u;
            float pt_x = target_pts[tgt_off + 0u];
            float pt_y = target_pts[tgt_off + 1u];
            float pt_z = target_pts[tgt_off + 2u];

            // C_sum = C_t + C_s
            float C_sum[9];
            uint sc_off = gid * 9u;
            uint tc_off = uint(target_idx) * 9u;
            for (int k = 0; k < 9; k++) {
                C_sum[k] = target_covs[tc_off + k] + source_covs[sc_off + k];
            }

            float Omega[9];
            if (invert3x3Sym(C_sum, Omega)) {
                float err[3];
                err[0] = pt_x - ps_x;
                err[1] = pt_y - ps_y;
                err[2] = pt_z - ps_z;

                float We[3];
                mul3(Omega, err, We);

                float x = ps_x, y = ps_y, z = ps_z;

                // b = [ps × We ; We]
                local_b[0] = y*We[2] - z*We[1];
                local_b[1] = z*We[0] - x*We[2];
                local_b[2] = x*We[1] - y*We[0];
                local_b[3] = We[0];
                local_b[4] = We[1];
                local_b[5] = We[2];

                // skew-symmetric columns of ps
                float s0[3]; s0[0] = 0.0; s0[1] =  -z; s0[2] =   y;
                float s1[3]; s1[0] =   z; s1[1] = 0.0; s1[2] =  -x;
                float s2[3]; s2[0] =  -y; s2[1] =   x; s2[2] = 0.0;

                float ws0[3], ws1[3], ws2[3];
                mul3(Omega, s0, ws0);
                mul3(Omega, s1, ws1);
                mul3(Omega, s2, ws2);

                // H_rr
                local_H[0*6+0] = y*ws0[2]-z*ws0[1]; local_H[0*6+1] = y*ws1[2]-z*ws1[1]; local_H[0*6+2] = y*ws2[2]-z*ws2[1];
                local_H[1*6+0] = z*ws0[0]-x*ws0[2]; local_H[1*6+1] = z*ws1[0]-x*ws1[2]; local_H[1*6+2] = z*ws2[0]-x*ws2[2];
                local_H[2*6+0] = x*ws0[1]-y*ws0[0]; local_H[2*6+1] = x*ws1[1]-y*ws1[0]; local_H[2*6+2] = x*ws2[1]-y*ws2[0];

                // H_rt
                local_H[0*6+3] = ws0[0]; local_H[0*6+4] = ws0[1]; local_H[0*6+5] = ws0[2];
                local_H[1*6+3] = ws1[0]; local_H[1*6+4] = ws1[1]; local_H[1*6+5] = ws1[2];
                local_H[2*6+3] = ws2[0]; local_H[2*6+4] = ws2[1]; local_H[2*6+5] = ws2[2];

                // H_tr = H_rt^T
                local_H[3*6+0] = local_H[0*6+3]; local_H[3*6+1] = local_H[1*6+3]; local_H[3*6+2] = local_H[2*6+3];
                local_H[4*6+0] = local_H[0*6+4]; local_H[4*6+1] = local_H[1*6+4]; local_H[4*6+2] = local_H[2*6+4];
                local_H[5*6+0] = local_H[0*6+5]; local_H[5*6+1] = local_H[1*6+5]; local_H[5*6+2] = local_H[2*6+5];

                // H_tt = Omega
                local_H[3*6+3] = Omega[0]; local_H[3*6+4] = Omega[1]; local_H[3*6+5] = Omega[2];
                local_H[4*6+3] = Omega[3]; local_H[4*6+4] = Omega[4]; local_H[4*6+5] = Omega[5];
                local_H[5*6+3] = Omega[6]; local_H[5*6+4] = Omega[7]; local_H[5*6+5] = Omega[8];
            }
        }
    }

    // 各スレッドの結果を shared memory へアトミック加算
    for (int i = 0; i < 36; i++) {
        if (local_H[i] != 0.0) atomicAddSharedH(uint(i), local_H[i]);
    }
    for (int i = 0; i < 6; i++) {
        if (local_b[i] != 0.0) atomicAddSharedB(uint(i), local_b[i]);
    }
    barrier();

    // shared memory → ワークグループ固有スロットへ単純 store（グローバルアトミック不要）
    // d_partial_H[wg_id * 36 + i],  d_partial_b[wg_id * 6 + i]
    barrier();
    uint wg_id = gl_WorkGroupID.x;
    if (lid < 36u) {
        d_partial_H[wg_id * 36u + lid] = uintBitsToFloat(s_H[lid]);
    }
    if (lid < 6u) {
        d_partial_b[wg_id * 6u + lid] = uintBitsToFloat(s_b[lid]);
    }
}