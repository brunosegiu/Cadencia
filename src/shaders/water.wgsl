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
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_type: u32,
};

struct WaterOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_depth: f32,
    @location(3) uv: vec2<f32>,
};

struct GBufferOut {
    @location(0) lit_color: vec4<f32>,
    @location(1) normal_id: vec4<f32>,
    @location(2) depth_meta: vec4<f32>,
};

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
    let biome_id = frame.fog_color.w; // packed? Wait: fog_color.w is fog_density! Let's check sun_dir.w (disc_strength) -> >0 means Coastal Ocean!

    var col = in.color.rgb;
    if (frame.sun_dir.w > 0.5) {
        // Coastal Sunset Ocean: calm slate-blue water + shimmering horizontal golden sun-reflection path
        let view_dir_xz = normalize(vec2<f32>(-V.x, -V.z));
        let sun_dir_xz = normalize(vec2<f32>(L.x, L.z));
        let align = max(dot(view_dir_xz, sun_dir_xz), 0.0);

        let ripple1 = sin(in.world_pos.z * 0.85 + in.world_pos.x * 0.25 + t * 1.1);
        let ripple2 = cos(in.world_pos.x * 0.55 - in.world_pos.z * 0.4 + t * 0.7);
        let wave = ripple1 * 0.5 + ripple2 * 0.5;

        // Warm horizon water blend
        let far_t = smoothstep(50.0, 620.0, in.view_depth);
        col = mix(vec3<f32>(0.37, 0.50, 0.60), vec3<f32>(0.76, 0.68, 0.58), far_t * 0.65);

        // Sparkling sunset path across water
        let sun_path = pow(align, 11.0);
        let glitter = smoothstep(0.10, 0.75, sun_path + wave * 0.32 * sun_path);
        col = mix(col, vec3<f32>(1.00, 0.93, 0.70), clamp(glitter * 0.96, 0.0, 1.0));

        // Subtle delicate horizontal comic water streaks
        let streak = smoothstep(0.88, 0.98, sin(in.world_pos.z * 1.4 + sin(in.world_pos.x * 0.2) * 2.0 + t * 0.5));
        col = mix(col, vec3<f32>(0.92, 0.86, 0.74), streak * 0.28 * (1.0 - far_t * 0.55));
    } else {
        // Mountain canyon silver-white winding river with sunlit specular sheen
        let ripple = sin(in.world_pos.z * 0.7 + in.world_pos.x * 0.4 + t * 0.8) * 0.5 + 0.5;
        col = mix(vec3<f32>(0.84, 0.88, 0.86), vec3<f32>(0.97, 0.96, 0.92), ripple * 0.65);
    }

    let _unused = biome_id;
    let fog_dist = max(in.view_depth - 45.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w * 0.75);
    col = mix(col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.72));

    let norm_d = clamp(in.view_depth / frame.sky_zenith.w, 0.0, 0.9999);
    let d_scaled = norm_d * 255.0;
    let d_hi = floor(d_scaled) / 255.0;
    let d_lo = fract(d_scaled);

    var out: GBufferOut;
    out.lit_color = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    // Flat upward normal, outline_weight = 0.0 on water interior so shore edges get inked by terrain border
    let vn = normalize((frame.view_mat * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    out.normal_id = vec4<f32>(vn * 0.5 + 0.5, obj.obj_info.x);
    out.depth_meta = vec4<f32>(d_hi, d_lo, 0.0, 0.0);
    return out;
}
