use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SceneVertex {
    pub pos: [f32; 3],       // 12 bytes (Float32x3, offset 0)
    pub normal_mat: [i8; 4], //  4 bytes (Snorm8x4,  offset 12: xyz = normal, w = material_type)
    pub uv: [u16; 2],        //  4 bytes (Float16x2, offset 16: IEEE-754 f16 UVs)
    pub color: [u8; 4],      //  4 bytes (Unorm8x4,  offset 20: RGBA8 vertex color)
}

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x007F_FFFF;
    if exp <= 0 {
        if exp < -10 {
            sign
        } else {
            let m = (mant | 0x0080_0000) >> (1 - exp + 13);
            sign | (m as u16)
        }
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

pub fn pack_snorm8(v: f32) -> i8 {
    (v.clamp(-1.0, 1.0) * 127.0).round() as i8
}

pub fn pack_unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl SceneVertex {
    pub fn new(pos: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4], material_type: u32) -> Self {
        Self {
            pos,
            normal_mat: [
                pack_snorm8(normal[0]),
                pack_snorm8(normal[1]),
                pack_snorm8(normal[2]),
                (material_type.min(127)) as i8,
            ],
            uv: [f32_to_f16(uv[0]), f32_to_f16(uv[1])],
            color: [
                pack_unorm8(color[0]),
                pack_unorm8(color[1]),
                pack_unorm8(color[2]),
                pack_unorm8(color[3]),
            ],
        }
    }

    pub fn new_packed_prop(
        pos: [f32; 3],
        normal: [f32; 3],
        uv: [u16; 2],
        rgb: [u8; 3],
        material_type: u8,
        object_id: u32,
        outline_weight: f32,
        has_crease: bool,
    ) -> Self {
        let outline_q3 = ((outline_weight.clamp(0.0, 1.0) * 7.0).round() as u8).min(7);
        let crease_bit = if has_crease { 1u8 } else { 0u8 };
        let packed_mat = (material_type & 7) | (outline_q3 << 3) | (crease_bit << 6);
        Self {
            pos,
            normal_mat: [
                pack_snorm8(normal[0]),
                pack_snorm8(normal[1]),
                pack_snorm8(normal[2]),
                packed_mat as i8,
            ],
            uv,
            color: [rgb[0], rgb[1], rgb[2], (object_id.min(255)) as u8],
        }
    }

    pub fn set_normal(&mut self, normal: [f32; 3]) {
        self.normal_mat[0] = pack_snorm8(normal[0]);
        self.normal_mat[1] = pack_snorm8(normal[1]);
        self.normal_mat[2] = pack_snorm8(normal[2]);
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress, // 24 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Snorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float16x2,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    BikeFrame,
    BikeWheel,
    BikeCrank,
    RiderTorso,
    RiderHeadHelmet,
    RiderBackpack,
    RiderThigh,
    RiderCalfShoe,
    FencePost,
    BoulderLarge,
    BoulderMedium,
    CliffOutcrop,
    BushCluster,
    WildflowerTuft,
    TreeCanopy,
    TreePine,
    RanchHouse,
    CondorBird,
    PalmTree,
    CoastalBuildingTall,
    CoastalBuildingMid,
    ColonialTownhouse,
    Flagpole,
    Streetlamp,
    PromenadeBench,
    Sailboat,
    Pedestrian,
    // Campo (Uruguayan Countryside) 10-variant rocks, trees, palms, bushes, and fauna
    CampoRockGranite(u8),
    CampoRockSlate(u8),
    CampoRockCluster(u8),
    TreeEucalyptus(u8),
    TreeOmbu(u8),
    TreeCeibo(u8),
    SmallPalm(u8),
    CeiboBush(u8),
    PampasCortadera(u8),
    ChircaBush(u8),
    CampoGrass(u8),
    Horse(u8),
    Cow(u8),
    Sheep(u8),
    Mulita(u8),
    // Rambla (Montevideo Coastal Promenade) 20 building types + walking & seated pedestrians (with mate)
    RamblaBuilding(u8),
    RamblaPedestrian(u8),
    RamblaSeated(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialRole {
    Default,
    BikeFrame,
    JerseyPrimary,
    JerseySecondary,
    JerseyAccent,
    Helmet,
    HelmetAccent,
    Shorts,
    Skin,
    Foliage,
    BuildingWall,
    WindowDark,
}

#[derive(Debug, Clone)]
pub struct MeshSection {
    pub vertices: Vec<SceneVertex>,
    pub indices: Vec<u32>,
    pub role: MaterialRole,
    pub base_color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub sections: Vec<MeshSection>,
    pub bounding_radius: f32,
}

pub struct AssetBank {
    pub models: HashMap<ModelId, LoadedModel>,
}

fn classify_role(id: ModelId, mat_name: &str) -> (MaterialRole, u32) {
    let n = mat_name.to_ascii_lowercase();
    if n.contains("bike_frame") {
        (MaterialRole::BikeFrame, 4)
    } else if n.contains("jersey_primary") {
        (MaterialRole::JerseyPrimary, 4)
    } else if n.contains("jersey_secondary") {
        (MaterialRole::JerseySecondary, 4)
    } else if n.contains("jersey_accent") {
        (MaterialRole::JerseyAccent, 4)
    } else if n.contains("helmet_dark") {
        (MaterialRole::Default, 4)
    } else if n.contains("helmet_accent") {
        (MaterialRole::HelmetAccent, 4)
    } else if n.contains("helmet") {
        (MaterialRole::Helmet, 4)
    } else if n.contains("shorts") {
        (MaterialRole::Shorts, 4)
    } else if n.contains("skin") {
        (MaterialRole::Skin, 4)
    } else if n.contains("foliage") || n.contains("frond") || n.contains("leaf") || n.contains("grass") {
        (MaterialRole::Foliage, 2)
    } else if n.contains("window") {
        (MaterialRole::WindowDark, 5)
    } else if n.contains("stucco") || n.contains("trim") {
        (MaterialRole::BuildingWall, 5)
    } else {
        let mat_type = match id {
            ModelId::BoulderLarge
            | ModelId::BoulderMedium
            | ModelId::CliffOutcrop
            | ModelId::CampoRockGranite(_)
            | ModelId::CampoRockSlate(_)
            | ModelId::CampoRockCluster(_) => 0,
            ModelId::RanchHouse
            | ModelId::CoastalBuildingTall
            | ModelId::CoastalBuildingMid
            | ModelId::ColonialTownhouse
            | ModelId::RamblaBuilding(_)
            | ModelId::FencePost
            | ModelId::PromenadeBench => 5,
            _ => 4,
        };
        (MaterialRole::Default, mat_type)
    }
}

fn linear_to_srgb_ch(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn load_glb(id: ModelId, bytes: &[u8]) -> LoadedModel {
    let gltf = gltf::Gltf::from_slice(bytes).expect("valid embedded GLB");
    let blob = gltf.blob.as_deref();
    let mut sections = Vec::new();

    for mesh in gltf.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| match buf.source() {
                gltf::buffer::Source::Bin => blob,
                gltf::buffer::Source::Uri(_) => None,
            });
            let Some(pos_iter) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = pos_iter.collect();
            let normals: Vec<[f32; 3]> = if let Some(n_iter) = reader.read_normals() {
                n_iter.collect()
            } else {
                vec![[0.0, 1.0, 0.0]; positions.len()]
            };
            let uvs: Vec<[f32; 2]> = if let Some(uv_iter) = reader.read_tex_coords(0) {
                uv_iter.into_f32().collect()
            } else {
                vec![[0.0, 0.0]; positions.len()]
            };
            let indices: Vec<u32> = if let Some(idx_iter) = reader.read_indices() {
                idx_iter.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };

            let mat = prim.material();
            let mat_name = mat.name().unwrap_or("");
            let pbr = mat.pbr_metallic_roughness().base_color_factor();
            // Convert linear glTF factor to display sRGB palette values
            let color = [
                linear_to_srgb_ch(pbr[0]),
                linear_to_srgb_ch(pbr[1]),
                linear_to_srgb_ch(pbr[2]),
                pbr[3],
            ];
            let (role, material_type) = classify_role(id, mat_name);
            let tintable_white = matches!(
                role,
                MaterialRole::BikeFrame
                    | MaterialRole::JerseyPrimary
                    | MaterialRole::JerseySecondary
                    | MaterialRole::JerseyAccent
                    | MaterialRole::Helmet
                    | MaterialRole::HelmetAccent
            );
            let vert_color = if tintable_white { [1.0, 1.0, 1.0, 1.0] } else { color };

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                vertices.push(SceneVertex::new(
                    positions[i],
                    normals[i],
                    uvs[i],
                    vert_color,
                    material_type,
                ));
            }

            // If normals were absent, compute geometric triangle normals
            if reader.read_normals().is_none() {
                for tri in indices.chunks_exact(3) {
                    let a = Vec3::from(vertices[tri[0] as usize].pos);
                    let b = Vec3::from(vertices[tri[1] as usize].pos);
                    let c = Vec3::from(vertices[tri[2] as usize].pos);
                    let n = (b - a).cross(c - a).normalize_or_zero();
                    for &idx in tri {
                        vertices[idx as usize].set_normal(n.into());
                    }
                }
            }

            sections.push(MeshSection {
                vertices,
                indices,
                role,
                base_color: color,
            });
        }
    }

    let mut min_pt = Vec3::splat(f32::MAX);
    let mut max_pt = Vec3::splat(f32::MIN);
    for sec in &sections {
        for v in &sec.vertices {
            let p = Vec3::from(v.pos);
            min_pt = min_pt.min(p);
            max_pt = max_pt.max(p);
        }
    }
    let bounding_radius = if min_pt.x <= max_pt.x {
        (0.5 * (max_pt - min_pt)).length().max(0.05)
    } else {
        0.5
    };

    LoadedModel {
        sections,
        bounding_radius,
    }
}

impl AssetBank {
    pub fn load_embedded() -> Self {
        let mut models = HashMap::new();
        let entries: &[(ModelId, &[u8])] = &[
            (ModelId::BikeFrame, include_bytes!("../assets/models/bike_frame.glb")),
            (ModelId::BikeWheel, include_bytes!("../assets/models/bike_wheel.glb")),
            (ModelId::BikeCrank, include_bytes!("../assets/models/bike_crank.glb")),
            (ModelId::RiderTorso, include_bytes!("../assets/models/rider_torso.glb")), // Manifold seated cyclist: aligned neck, flat athletic back, balanced glutes, sculpted wrists & UV leg weights
            (ModelId::RiderHeadHelmet, include_bytes!("../assets/models/rider_head_helmet.glb")), // Sculpted chin & recessed skull-base neck socket
            (ModelId::RiderBackpack, include_bytes!("../assets/models/rider_backpack.glb")),
            (ModelId::RiderThigh, include_bytes!("../assets/models/rider_thigh.glb")),
            (ModelId::RiderCalfShoe, include_bytes!("../assets/models/rider_calf_shoe.glb")),
            (ModelId::FencePost, include_bytes!("../assets/models/fence_post.glb")),
            (ModelId::BoulderLarge, include_bytes!("../assets/models/boulder_large.glb")),
            (ModelId::BoulderMedium, include_bytes!("../assets/models/boulder_medium.glb")),
            (ModelId::CliffOutcrop, include_bytes!("../assets/models/cliff_outcrop.glb")),
            (ModelId::BushCluster, include_bytes!("../assets/models/bush_cluster.glb")),
            (ModelId::WildflowerTuft, include_bytes!("../assets/models/wildflower_tuft.glb")),
            (ModelId::TreeCanopy, include_bytes!("../assets/models/tree_canopy.glb")),
            (ModelId::TreePine, include_bytes!("../assets/models/tree_pine.glb")),
            (ModelId::RanchHouse, include_bytes!("../assets/models/ranch_house.glb")),
            (ModelId::CondorBird, include_bytes!("../assets/models/condor_bird.glb")),
            (ModelId::PalmTree, include_bytes!("../assets/models/palm_tree.glb")),
            (ModelId::CoastalBuildingTall, include_bytes!("../assets/models/coastal_building_tall.glb")),
            (ModelId::CoastalBuildingMid, include_bytes!("../assets/models/coastal_building_mid.glb")),
            (ModelId::ColonialTownhouse, include_bytes!("../assets/models/colonial_townhouse.glb")),
            (ModelId::Flagpole, include_bytes!("../assets/models/flagpole.glb")),
            (ModelId::Streetlamp, include_bytes!("../assets/models/streetlamp.glb")),
            (ModelId::PromenadeBench, include_bytes!("../assets/models/promenade_bench.glb")),
            (ModelId::Sailboat, include_bytes!("../assets/models/sailboat.glb")),
            (ModelId::Pedestrian, include_bytes!("../assets/models/pedestrian.glb")),
            // Campo Granite Rock variants (10)
            (ModelId::CampoRockGranite(0), include_bytes!("../assets/models/campo_rock_granite_0.glb")),
            (ModelId::CampoRockGranite(1), include_bytes!("../assets/models/campo_rock_granite_1.glb")),
            (ModelId::CampoRockGranite(2), include_bytes!("../assets/models/campo_rock_granite_2.glb")),
            (ModelId::CampoRockGranite(3), include_bytes!("../assets/models/campo_rock_granite_3.glb")),
            (ModelId::CampoRockGranite(4), include_bytes!("../assets/models/campo_rock_granite_4.glb")),
            (ModelId::CampoRockGranite(5), include_bytes!("../assets/models/campo_rock_granite_5.glb")),
            (ModelId::CampoRockGranite(6), include_bytes!("../assets/models/campo_rock_granite_6.glb")),
            (ModelId::CampoRockGranite(7), include_bytes!("../assets/models/campo_rock_granite_7.glb")),
            (ModelId::CampoRockGranite(8), include_bytes!("../assets/models/campo_rock_granite_8.glb")),
            (ModelId::CampoRockGranite(9), include_bytes!("../assets/models/campo_rock_granite_9.glb")),
            // Campo Slate / Laja Rock variants (10)
            (ModelId::CampoRockSlate(0), include_bytes!("../assets/models/campo_rock_slate_0.glb")),
            (ModelId::CampoRockSlate(1), include_bytes!("../assets/models/campo_rock_slate_1.glb")),
            (ModelId::CampoRockSlate(2), include_bytes!("../assets/models/campo_rock_slate_2.glb")),
            (ModelId::CampoRockSlate(3), include_bytes!("../assets/models/campo_rock_slate_3.glb")),
            (ModelId::CampoRockSlate(4), include_bytes!("../assets/models/campo_rock_slate_4.glb")),
            (ModelId::CampoRockSlate(5), include_bytes!("../assets/models/campo_rock_slate_5.glb")),
            (ModelId::CampoRockSlate(6), include_bytes!("../assets/models/campo_rock_slate_6.glb")),
            (ModelId::CampoRockSlate(7), include_bytes!("../assets/models/campo_rock_slate_7.glb")),
            (ModelId::CampoRockSlate(8), include_bytes!("../assets/models/campo_rock_slate_8.glb")),
            (ModelId::CampoRockSlate(9), include_bytes!("../assets/models/campo_rock_slate_9.glb")),
            // Campo Multi-Boulder Cluster Rock variants (10)
            (ModelId::CampoRockCluster(0), include_bytes!("../assets/models/campo_rock_cluster_0.glb")),
            (ModelId::CampoRockCluster(1), include_bytes!("../assets/models/campo_rock_cluster_1.glb")),
            (ModelId::CampoRockCluster(2), include_bytes!("../assets/models/campo_rock_cluster_2.glb")),
            (ModelId::CampoRockCluster(3), include_bytes!("../assets/models/campo_rock_cluster_3.glb")),
            (ModelId::CampoRockCluster(4), include_bytes!("../assets/models/campo_rock_cluster_4.glb")),
            (ModelId::CampoRockCluster(5), include_bytes!("../assets/models/campo_rock_cluster_5.glb")),
            (ModelId::CampoRockCluster(6), include_bytes!("../assets/models/campo_rock_cluster_6.glb")),
            (ModelId::CampoRockCluster(7), include_bytes!("../assets/models/campo_rock_cluster_7.glb")),
            (ModelId::CampoRockCluster(8), include_bytes!("../assets/models/campo_rock_cluster_8.glb")),
            (ModelId::CampoRockCluster(9), include_bytes!("../assets/models/campo_rock_cluster_9.glb")),
            // Tall Eucalyptus Tree variants (10)
            (ModelId::TreeEucalyptus(0), include_bytes!("../assets/models/tree_eucalyptus_0.glb")),
            (ModelId::TreeEucalyptus(1), include_bytes!("../assets/models/tree_eucalyptus_1.glb")),
            (ModelId::TreeEucalyptus(2), include_bytes!("../assets/models/tree_eucalyptus_2.glb")),
            (ModelId::TreeEucalyptus(3), include_bytes!("../assets/models/tree_eucalyptus_3.glb")),
            (ModelId::TreeEucalyptus(4), include_bytes!("../assets/models/tree_eucalyptus_4.glb")),
            (ModelId::TreeEucalyptus(5), include_bytes!("../assets/models/tree_eucalyptus_5.glb")),
            (ModelId::TreeEucalyptus(6), include_bytes!("../assets/models/tree_eucalyptus_6.glb")),
            (ModelId::TreeEucalyptus(7), include_bytes!("../assets/models/tree_eucalyptus_7.glb")),
            (ModelId::TreeEucalyptus(8), include_bytes!("../assets/models/tree_eucalyptus_8.glb")),
            (ModelId::TreeEucalyptus(9), include_bytes!("../assets/models/tree_eucalyptus_9.glb")),
            // Spreading Ombu Tree variants (10)
            (ModelId::TreeOmbu(0), include_bytes!("../assets/models/tree_ombu_0.glb")),
            (ModelId::TreeOmbu(1), include_bytes!("../assets/models/tree_ombu_1.glb")),
            (ModelId::TreeOmbu(2), include_bytes!("../assets/models/tree_ombu_2.glb")),
            (ModelId::TreeOmbu(3), include_bytes!("../assets/models/tree_ombu_3.glb")),
            (ModelId::TreeOmbu(4), include_bytes!("../assets/models/tree_ombu_4.glb")),
            (ModelId::TreeOmbu(5), include_bytes!("../assets/models/tree_ombu_5.glb")),
            (ModelId::TreeOmbu(6), include_bytes!("../assets/models/tree_ombu_6.glb")),
            (ModelId::TreeOmbu(7), include_bytes!("../assets/models/tree_ombu_7.glb")),
            (ModelId::TreeOmbu(8), include_bytes!("../assets/models/tree_ombu_8.glb")),
            (ModelId::TreeOmbu(9), include_bytes!("../assets/models/tree_ombu_9.glb")),
            // Ceibo / Espinillo Countryside Tree variants (10)
            (ModelId::TreeCeibo(0), include_bytes!("../assets/models/tree_ceibo_0.glb")),
            (ModelId::TreeCeibo(1), include_bytes!("../assets/models/tree_ceibo_1.glb")),
            (ModelId::TreeCeibo(2), include_bytes!("../assets/models/tree_ceibo_2.glb")),
            (ModelId::TreeCeibo(3), include_bytes!("../assets/models/tree_ceibo_3.glb")),
            (ModelId::TreeCeibo(4), include_bytes!("../assets/models/tree_ceibo_4.glb")),
            (ModelId::TreeCeibo(5), include_bytes!("../assets/models/tree_ceibo_5.glb")),
            (ModelId::TreeCeibo(6), include_bytes!("../assets/models/tree_ceibo_6.glb")),
            (ModelId::TreeCeibo(7), include_bytes!("../assets/models/tree_ceibo_7.glb")),
            (ModelId::TreeCeibo(8), include_bytes!("../assets/models/tree_ceibo_8.glb")),
            (ModelId::TreeCeibo(9), include_bytes!("../assets/models/tree_ceibo_9.glb")),
            // Small Butia / Caranday Palm variants (10)
            (ModelId::SmallPalm(0), include_bytes!("../assets/models/small_palm_0.glb")),
            (ModelId::SmallPalm(1), include_bytes!("../assets/models/small_palm_1.glb")),
            (ModelId::SmallPalm(2), include_bytes!("../assets/models/small_palm_2.glb")),
            (ModelId::SmallPalm(3), include_bytes!("../assets/models/small_palm_3.glb")),
            (ModelId::SmallPalm(4), include_bytes!("../assets/models/small_palm_4.glb")),
            (ModelId::SmallPalm(5), include_bytes!("../assets/models/small_palm_5.glb")),
            (ModelId::SmallPalm(6), include_bytes!("../assets/models/small_palm_6.glb")),
            (ModelId::SmallPalm(7), include_bytes!("../assets/models/small_palm_7.glb")),
            (ModelId::SmallPalm(8), include_bytes!("../assets/models/small_palm_8.glb")),
            (ModelId::SmallPalm(9), include_bytes!("../assets/models/small_palm_9.glb")),
            // Ceibo Red-Flowering Scrub Bush variants (10)
            (ModelId::CeiboBush(0), include_bytes!("../assets/models/ceibo_bush_0.glb")),
            (ModelId::CeiboBush(1), include_bytes!("../assets/models/ceibo_bush_1.glb")),
            (ModelId::CeiboBush(2), include_bytes!("../assets/models/ceibo_bush_2.glb")),
            (ModelId::CeiboBush(3), include_bytes!("../assets/models/ceibo_bush_3.glb")),
            (ModelId::CeiboBush(4), include_bytes!("../assets/models/ceibo_bush_4.glb")),
            (ModelId::CeiboBush(5), include_bytes!("../assets/models/ceibo_bush_5.glb")),
            (ModelId::CeiboBush(6), include_bytes!("../assets/models/ceibo_bush_6.glb")),
            (ModelId::CeiboBush(7), include_bytes!("../assets/models/ceibo_bush_7.glb")),
            (ModelId::CeiboBush(8), include_bytes!("../assets/models/ceibo_bush_8.glb")),
            (ModelId::CeiboBush(9), include_bytes!("../assets/models/ceibo_bush_9.glb")),
            // Pampas Cortadera White-Plume Grass variants (10, Blender MCP authored)
            (ModelId::PampasCortadera(0), include_bytes!("../assets/models/pampas_cortadera_0.glb")),
            (ModelId::PampasCortadera(1), include_bytes!("../assets/models/pampas_cortadera_1.glb")),
            (ModelId::PampasCortadera(2), include_bytes!("../assets/models/pampas_cortadera_2.glb")),
            (ModelId::PampasCortadera(3), include_bytes!("../assets/models/pampas_cortadera_3.glb")),
            (ModelId::PampasCortadera(4), include_bytes!("../assets/models/pampas_cortadera_4.glb")),
            (ModelId::PampasCortadera(5), include_bytes!("../assets/models/pampas_cortadera_5.glb")),
            (ModelId::PampasCortadera(6), include_bytes!("../assets/models/pampas_cortadera_6.glb")),
            (ModelId::PampasCortadera(7), include_bytes!("../assets/models/pampas_cortadera_7.glb")),
            (ModelId::PampasCortadera(8), include_bytes!("../assets/models/pampas_cortadera_8.glb")),
            (ModelId::PampasCortadera(9), include_bytes!("../assets/models/pampas_cortadera_9.glb")),
            // Chirca Countryside Scrub Bush (V-shaped broom shrub) variants (10)
            (ModelId::ChircaBush(0), include_bytes!("../assets/models/chirca_bush_0.glb")),
            (ModelId::ChircaBush(1), include_bytes!("../assets/models/chirca_bush_1.glb")),
            (ModelId::ChircaBush(2), include_bytes!("../assets/models/chirca_bush_2.glb")),
            (ModelId::ChircaBush(3), include_bytes!("../assets/models/chirca_bush_3.glb")),
            (ModelId::ChircaBush(4), include_bytes!("../assets/models/chirca_bush_4.glb")),
            (ModelId::ChircaBush(5), include_bytes!("../assets/models/chirca_bush_5.glb")),
            (ModelId::ChircaBush(6), include_bytes!("../assets/models/chirca_bush_6.glb")),
            (ModelId::ChircaBush(7), include_bytes!("../assets/models/chirca_bush_7.glb")),
            (ModelId::ChircaBush(8), include_bytes!("../assets/models/chirca_bush_8.glb")),
            (ModelId::ChircaBush(9), include_bytes!("../assets/models/chirca_bush_9.glb")),
            // Campo Pasture Grass Clump variants (10)
            (ModelId::CampoGrass(0), include_bytes!("../assets/models/campo_grass_0.glb")),
            (ModelId::CampoGrass(1), include_bytes!("../assets/models/campo_grass_1.glb")),
            (ModelId::CampoGrass(2), include_bytes!("../assets/models/campo_grass_2.glb")),
            (ModelId::CampoGrass(3), include_bytes!("../assets/models/campo_grass_3.glb")),
            (ModelId::CampoGrass(4), include_bytes!("../assets/models/campo_grass_4.glb")),
            (ModelId::CampoGrass(5), include_bytes!("../assets/models/campo_grass_5.glb")),
            (ModelId::CampoGrass(6), include_bytes!("../assets/models/campo_grass_6.glb")),
            (ModelId::CampoGrass(7), include_bytes!("../assets/models/campo_grass_7.glb")),
            (ModelId::CampoGrass(8), include_bytes!("../assets/models/campo_grass_8.glb")),
            (ModelId::CampoGrass(9), include_bytes!("../assets/models/campo_grass_9.glb")),
            // Campo Skinned Quadruped Animals (canonical rest-pose GLBs): Horses, Cows, Sheep, Mulitas
            (ModelId::Horse(0), include_bytes!("../assets/models/horse_0.glb")),
            (ModelId::Horse(1), include_bytes!("../assets/models/horse_1.glb")),
            (ModelId::Horse(2), include_bytes!("../assets/models/horse_2.glb")),
            (ModelId::Horse(3), include_bytes!("../assets/models/horse_3.glb")),
            (ModelId::Cow(0), include_bytes!("../assets/models/cow_0.glb")),
            (ModelId::Cow(1), include_bytes!("../assets/models/cow_1.glb")),
            (ModelId::Cow(2), include_bytes!("../assets/models/cow_2.glb")),
            (ModelId::Cow(3), include_bytes!("../assets/models/cow_3.glb")),
            (ModelId::Sheep(0), include_bytes!("../assets/models/sheep_0.glb")),
            (ModelId::Sheep(1), include_bytes!("../assets/models/sheep_1.glb")),
            (ModelId::Sheep(2), include_bytes!("../assets/models/sheep_2.glb")),
            (ModelId::Sheep(3), include_bytes!("../assets/models/sheep_3.glb")),
            (ModelId::Mulita(0), include_bytes!("../assets/models/mulita_0.glb")),
            (ModelId::Mulita(1), include_bytes!("../assets/models/mulita_1.glb")),
            (ModelId::Mulita(2), include_bytes!("../assets/models/mulita_2.glb")),
            (ModelId::Mulita(3), include_bytes!("../assets/models/mulita_3.glb")),
            // Rambla 20 distinct Montevideo coastal architectural building types (Blender MCP authored)
            (ModelId::RamblaBuilding(0), include_bytes!("../assets/models/rambla_building_0.glb")),
            (ModelId::RamblaBuilding(1), include_bytes!("../assets/models/rambla_building_1.glb")),
            (ModelId::RamblaBuilding(2), include_bytes!("../assets/models/rambla_building_2.glb")),
            (ModelId::RamblaBuilding(3), include_bytes!("../assets/models/rambla_building_3.glb")),
            (ModelId::RamblaBuilding(4), include_bytes!("../assets/models/rambla_building_4.glb")),
            (ModelId::RamblaBuilding(5), include_bytes!("../assets/models/rambla_building_5.glb")),
            (ModelId::RamblaBuilding(6), include_bytes!("../assets/models/rambla_building_6.glb")),
            (ModelId::RamblaBuilding(7), include_bytes!("../assets/models/rambla_building_7.glb")),
            (ModelId::RamblaBuilding(8), include_bytes!("../assets/models/rambla_building_8.glb")),
            (ModelId::RamblaBuilding(9), include_bytes!("../assets/models/rambla_building_9.glb")),
            (ModelId::RamblaBuilding(10), include_bytes!("../assets/models/rambla_building_10.glb")),
            (ModelId::RamblaBuilding(11), include_bytes!("../assets/models/rambla_building_11.glb")),
            (ModelId::RamblaBuilding(12), include_bytes!("../assets/models/rambla_building_12.glb")),
            (ModelId::RamblaBuilding(13), include_bytes!("../assets/models/rambla_building_13.glb")),
            (ModelId::RamblaBuilding(14), include_bytes!("../assets/models/rambla_building_14.glb")),
            (ModelId::RamblaBuilding(15), include_bytes!("../assets/models/rambla_building_15.glb")),
            (ModelId::RamblaBuilding(16), include_bytes!("../assets/models/rambla_building_16.glb")),
            (ModelId::RamblaBuilding(17), include_bytes!("../assets/models/rambla_building_17.glb")),
            (ModelId::RamblaBuilding(18), include_bytes!("../assets/models/rambla_building_18.glb")),
            (ModelId::RamblaBuilding(19), include_bytes!("../assets/models/rambla_building_19.glb")),
            // Rambla Walking & Bench-Seated Pedestrians (derived from cyclist base mesh, with termo & mate)
            (ModelId::RamblaPedestrian(0), include_bytes!("../assets/models/rambla_pedestrian_0.glb")),
            (ModelId::RamblaPedestrian(1), include_bytes!("../assets/models/rambla_pedestrian_1.glb")),
            (ModelId::RamblaPedestrian(2), include_bytes!("../assets/models/rambla_pedestrian_2.glb")),
            (ModelId::RamblaPedestrian(3), include_bytes!("../assets/models/rambla_pedestrian_3.glb")),
            (ModelId::RamblaSeated(0), include_bytes!("../assets/models/rambla_seated_0.glb")),
            (ModelId::RamblaSeated(1), include_bytes!("../assets/models/rambla_seated_1.glb")),
            (ModelId::RamblaSeated(2), include_bytes!("../assets/models/rambla_seated_2.glb")),
            (ModelId::RamblaSeated(3), include_bytes!("../assets/models/rambla_seated_3.glb")),
        ];
        for &(id, bytes) in entries {
            models.insert(id, load_glb(id, bytes));
        }
        Self { models }
    }

    pub fn get(&self, id: ModelId) -> &LoadedModel {
        &self.models[&id]
    }
}

#[allow(dead_code)]
fn _keep_vec_types(_: Vec2, _: Vec4) {}
