struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_mat: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    screen_params: vec4<f32>,
    outline_params: vec4<f32>,
    light_view_proj_c1: mat4x4<f32>,
    light_view_proj_c2: mat4x4<f32>,
    cascade_splits: vec4<f32>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>,
    skin_pose_r: vec4<f32>,
    skin_pose_l: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal_mat: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct WaterOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_depth: f32,
    @location(3) uv: vec2<f32>,
};

struct GBufferOut {
    @location(0) lit_color: vec4<f32>, // rgb = lit color, a = outline_weight
    @location(1) normal_id: vec4<f32>, // rg = octahedral view normal, b = crease_sensitivity, a = object_id
};

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-5));
    let res = select(
        (1.0 - abs(p.yx)) * vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0)),
        p,
        n.z >= 0.0
    );
    return res * 0.5 + 0.5;
}

@vertex
fn vs_water(in: VertexInput) -> WaterOut {
    let wp = obj.model * vec4<f32>(in.pos, 1.0);
    let vp = frame.view_mat * wp;
    var out: WaterOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.color = in.color;
    out.view_depth = -vp.z;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_water(in: WaterOut) -> GBufferOut {
    let V = normalize(frame.camera_pos.xyz - in.world_pos);
    let L = normalize(frame.sun_dir.xyz);
    let t = frame.camera_pos.w;

    var col = in.color.rgb;
    if (frame.sun_dir.w > 0.5) {
        // Coastal Sunset Ocean: calm slate-blue water + shimmering horizontal golden sun-reflection path
        let view_dir_xz = normalize(vec2<f32>(-V.x, -V.z));
        let sun_dir_xz = normalize(vec2<f32>(L.x, L.z));
        let align = max(dot(view_dir_xz, sun_dir_xz), 0.0);

        let ripple1 = sin(in.world_pos.z * 0.85 + in.world_pos.x * 0.25 + t * 1.1);
        let ripple2 = cos(in.world_pos.x * 0.55 - in.world_pos.z * 0.4 + t * 0.7);
        let wave = ripple1 * 0.5 + ripple2 * 0.5;

        let far_t = smoothstep(50.0, 620.0, in.view_depth);
        col = mix(vec3<f32>(0.37, 0.50, 0.60), vec3<f32>(0.76, 0.68, 0.58), far_t * 0.65);

        let sun_path = pow(align, 11.0);
        let glitter = smoothstep(0.10, 0.75, sun_path + wave * 0.32 * sun_path);
        col = mix(col, vec3<f32>(1.00, 0.93, 0.70), clamp(glitter * 0.96, 0.0, 1.0));

        let streak = smoothstep(0.88, 0.98, sin(in.world_pos.z * 1.4 + sin(in.world_pos.x * 0.2) * 2.0 + t * 0.5));
        col = mix(col, vec3<f32>(0.92, 0.86, 0.74), streak * 0.28 * (1.0 - far_t * 0.55));
    } else {
        // Mountain canyon silver-white winding river with sunlit specular sheen
        let ripple = sin(in.world_pos.z * 0.7 + in.world_pos.x * 0.4 + t * 0.8) * 0.5 + 0.5;
        col = mix(vec3<f32>(0.84, 0.88, 0.86), vec3<f32>(0.97, 0.96, 0.92), ripple * 0.65);
    }

    let fog_dist = max(in.view_depth - 45.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w * 0.75);
    col = mix(col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.72));

    var out: GBufferOut;
    // outline_weight = 0.0 in alpha on water interior so shore edges get inked by terrain border
    out.lit_color = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
    let vn = normalize((frame.view_mat * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    let oct_n = oct_encode(vn);
    out.normal_id = vec4<f32>(oct_n.x, oct_n.y, 0.0, obj.obj_info.x);
    return out;
}
