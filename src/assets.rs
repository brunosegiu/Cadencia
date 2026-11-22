use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SceneVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub material_type: u32,
}

impl SceneVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialRole {
    Default,
    BikeFrame,
    JerseyPrimary,
    JerseySecondary,
    Helmet,
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
    } else if n.contains("helmet") {
        (MaterialRole::Helmet, 4)
    } else if n.contains("shorts") {
        (MaterialRole::Shorts, 4)
    } else if n.contains("skin") {
        (MaterialRole::Skin, 4)
    } else if n.contains("foliage") || n.contains("frond") {
        (MaterialRole::Foliage, 2)
    } else if n.contains("window") {
        (MaterialRole::WindowDark, 5)
    } else if n.contains("stucco") || n.contains("trim") {
        (MaterialRole::BuildingWall, 5)
    } else {
        let mat_type = match id {
            ModelId::BoulderLarge | ModelId::BoulderMedium | ModelId::CliffOutcrop => 0,
            ModelId::RanchHouse
            | ModelId::CoastalBuildingTall
            | ModelId::CoastalBuildingMid
            | ModelId::ColonialTownhouse
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
                    | MaterialRole::Helmet
            );
            let vert_color = if tintable_white { [1.0, 1.0, 1.0, 1.0] } else { color };

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                vertices.push(SceneVertex {
                    pos: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                    color: vert_color,
                    material_type,
                });
            }

            // If normals were absent, compute geometric triangle normals
            if reader.read_normals().is_none() {
                for tri in indices.chunks_exact(3) {
                    let a = Vec3::from(vertices[tri[0] as usize].pos);
                    let b = Vec3::from(vertices[tri[1] as usize].pos);
                    let c = Vec3::from(vertices[tri[2] as usize].pos);
                    let n = (b - a).cross(c - a).normalize_or_zero();
                    for &idx in tri {
                        vertices[idx as usize].normal = n.into();
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

    LoadedModel { sections }
}

impl AssetBank {
    pub fn load_embedded() -> Self {
        let mut models = HashMap::new();
        let entries: &[(ModelId, &[u8])] = &[
            (ModelId::BikeFrame, include_bytes!("../assets/models/bike_frame.glb")),
            (ModelId::BikeWheel, include_bytes!("../assets/models/bike_wheel.glb")),
            (ModelId::BikeCrank, include_bytes!("../assets/models/bike_crank.glb")),
            (ModelId::RiderTorso, include_bytes!("../assets/models/rider_torso.glb")),
            (ModelId::RiderHeadHelmet, include_bytes!("../assets/models/rider_head_helmet.glb")),
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
