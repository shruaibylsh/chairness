import bpy
import os
import math
import random
import glob
import sys
import argparse
from mathutils import Vector
import addon_utils

def setup_scene():
    """Set up the Blender scene with lighting and camera."""
    # Clear existing objects: deselect and delete meshes, lights, and cameras.
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Create a camera
    bpy.ops.object.camera_add(location=(0, -3, 1.5), rotation=(math.radians(75), 0, 0))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    # Create a three-point lighting setup
    bpy.ops.object.light_add(type='AREA', radius=3, location=(3, -2, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 500
    key_light.data.color = (1.0, 0.95, 0.9)

    bpy.ops.object.light_add(type='AREA', radius=2, location=(-3, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 300
    fill_light.data.color = (0.9, 0.95, 1.0)

    bpy.ops.object.light_add(type='AREA', radius=2, location=(0, 3, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 400
    rim_light.data.color = (1.0, 1.0, 1.0)

    # Set up a transparent background using nodes
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    node_tree = world.node_tree
    # Clear default nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    bg_node = node_tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1, 1, 1, 0)
    bg_node.inputs['Strength'].default_value = 1.0

    output_node = node_tree.nodes.new('ShaderNodeOutputWorld')
    node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    return camera

def setup_render_settings(resolution=512, samples=64):
    """Configure render settings for optimal quality and speed."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Configure GPU rendering settings if available.
    if bpy.app.version >= (3, 0, 0):
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons.get('cycles')
        if cycles_prefs:
            cycles_preferences = cycles_prefs.preferences
            available_devices = [item.identifier for item in cycles_preferences.bl_rna.properties['compute_device_type'].enum_items]
            if 'METAL' in available_devices:
                cycles_preferences.compute_device_type = 'METAL'
            elif 'CUDA' in available_devices:
                cycles_preferences.compute_device_type = 'CUDA'
            elif 'OPTIX' in available_devices:
                cycles_preferences.compute_device_type = 'OPTIX'
            elif 'HIP' in available_devices:
                cycles_preferences.compute_device_type = 'HIP'
            for device in cycles_preferences.devices:
                device.use = True

    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    # Set output format to PNG with alpha (RGBA, 16-bit)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    # Enable denoising and adaptive sampling
    scene.cycles.use_denoising = True
    scene.cycles.adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

def import_chair_model(filepath):
    """Import a chair model from the given file path."""
    # Clear existing models
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.fbx': # recommended
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == '.blend':
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    elif ext in ['.gltf', '.glb']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Get all imported mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        raise ValueError(f"No mesh objects found in: {filepath}")

    # Select and join mesh objects if there is more than one
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    chair = bpy.context.active_object

    # Normalize the size of the chair model
    dimensions = chair.dimensions
    max_dim = max(dimensions.x, dimensions.y, dimensions.z)
    scale_factor = 2.0 / max_dim
    chair.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Center the model on the ground
    bbox_corners = [chair.matrix_world @ Vector(corner) for corner in chair.bound_box]
    min_z = min(corner.z for corner in bbox_corners)
    chair.location.z -= min_z

    # Collect existing materials
    materials = [slot.material for slot in chair.material_slots if slot.material]

    return chair, materials

def apply_textures_to_material(mat, textures_dir):
    """Modify the material node tree to use texture images if available."""
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Locate the Principled BSDF node; create one if necessary.
    principled = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
            break
    if principled is None:
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        output_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                break
        if output_node is None:
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            output_node.location = (400, 0)
        links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])

    # List available texture files from the directory.
    texture_files = os.listdir(textures_dir)

    def find_texture(keyword):
        for f in texture_files:
            if keyword.lower() in f.lower():
                return os.path.join(textures_dir, f)
        return None

    # Base Color (Albedo)
    base_color_path = find_texture('albedo') or find_texture('diffuse') or find_texture('basecolor')
    if base_color_path:
        base_color_node = nodes.new(type='ShaderNodeTexImage')
        try:
            base_color_node.image = bpy.data.images.load(base_color_path, check_existing=True)
        except Exception as e:
            print(f"Error loading base color texture: {str(e)}")
        base_color_node.location = (-600, 300)
        links.new(base_color_node.outputs['Color'], principled.inputs['Base Color'])

    # Roughness
    roughness_path = find_texture('roughness')
    if roughness_path:
        roughness_node = nodes.new(type='ShaderNodeTexImage')
        try:
            roughness_node.image = bpy.data.images.load(roughness_path, check_existing=True)
            roughness_node.image.colorspace_settings.name = 'Non-Color'
        except Exception as e:
            print(f"Error loading roughness texture: {str(e)}")
        roughness_node.location = (-600, 100)
        links.new(roughness_node.outputs['Color'], principled.inputs['Roughness'])

    # Metallic
    metallic_path = find_texture('metallic')
    if metallic_path:
        metallic_node = nodes.new(type='ShaderNodeTexImage')
        try:
            metallic_node.image = bpy.data.images.load(metallic_path, check_existing=True)
            metallic_node.image.colorspace_settings.name = 'Non-Color'
        except Exception as e:
            print(f"Error loading metallic texture: {str(e)}")
        metallic_node.location = (-600, -100)
        links.new(metallic_node.outputs['Color'], principled.inputs['Metallic'])

    # Normal Map
    normal_path = find_texture('normal')
    if normal_path:
        normal_tex_node = nodes.new(type='ShaderNodeTexImage')
        try:
            normal_tex_node.image = bpy.data.images.load(normal_path, check_existing=True)
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
        except Exception as e:
            print(f"Error loading normal texture: {str(e)}")
        normal_tex_node.location = (-600, -300)
        normal_map_node = nodes.new(type='ShaderNodeNormalMap')
        normal_map_node.location = (-300, -300)
        links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
        links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])

    return mat

def create_material_variations(materials, textures_dir=None):
    """
    Create material variations. If textures_dir is provided, the textures
    found there will be applied to the material nodes.
    """
    variations = []
    colors = [
        (0.8, 0.4, 0.2, 1.0),  # Wood - Medium
        (0.6, 0.3, 0.1, 1.0),  # Wood - Dark
        (0.9, 0.7, 0.5, 1.0),  # Wood - Light
        (0.1, 0.1, 0.1, 1.0),  # Black
        (0.8, 0.8, 0.8, 1.0),  # Light Grey
        (0.3, 0.3, 0.3, 1.0),  # Dark Grey
        (0.1, 0.2, 0.7, 1.0),  # Blue
        (0.7, 0.1, 0.1, 1.0),  # Red
        (0.1, 0.5, 0.1, 1.0),  # Green
    ]

    if not materials:
        # Create a default material if none exist.
        mat = bpy.data.materials.new("Chair_Material")
        mat.use_nodes = True
        bsdf = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break
        if bsdf:
            bsdf.inputs["Base Color"].default_value = random.choice(colors)
            bsdf.inputs["Roughness"].default_value = 0.3
            if "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = 0.2
        if textures_dir:
            mat = apply_textures_to_material(mat, textures_dir)
        variations.append(mat)
    else:
        # Create variations from existing materials.
        for base_mat in materials:
            for j in range(3):  # Generate three variations per material.
                new_mat = base_mat.copy()
                new_mat.name = f"{base_mat.name}_var{j}"
                if new_mat.use_nodes:
                    for node in new_mat.node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            if j > 0:
                                node.inputs["Base Color"].default_value = random.choice(colors)
                            node.inputs["Roughness"].default_value = random.uniform(0.1, 0.9)
                            node.inputs["Metallic"].default_value = random.uniform(0, 1) if random.random() > 0.7 else 0
                if textures_dir:
                    new_mat = apply_textures_to_material(new_mat, textures_dir)
                variations.append(new_mat)
    return variations

def apply_material(chair, material):
    """Apply the given material to all material slots of the chair."""
    for slot in chair.material_slots:
        slot.material = material

def render_chair_angles(chair, output_path, num_angles=12, elevation_angles=3):
    """Render the chair from multiple angles and elevations."""
    camera = bpy.context.scene.camera
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, chair.dimensions.z / 2))
    target = bpy.context.active_object
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    camera_dist = max(chair.dimensions.x, chair.dimensions.y, chair.dimensions.z) * 2.5
    if elevation_angles == 3:
        elev_list = [15, 30, 60]
    elif elevation_angles == 2:
        elev_list = [20, 45]
    else:
        elev_list = [30]
    renders_count = 0
    for elev_idx, elev in enumerate(elev_list):
        for angle_idx in range(num_angles):
            angle = 2 * math.pi * angle_idx / num_angles
            x = camera_dist * math.sin(angle)
            y = camera_dist * math.cos(angle)
            z = camera_dist * math.sin(math.radians(elev))
            camera.location = (x, y, z)
            bpy.context.view_layer.update()
            output_file = os.path.join(output_path, f"angle_{angle_idx:02d}_elev_{elev_idx:02d}.png")
            bpy.context.scene.render.filepath = output_file
            print(f"Rendering angle {angle_idx+1}/{num_angles}, elevation {elev_idx+1}/{len(elev_list)}")
            bpy.ops.render.render(write_still=True)
            renders_count += 1
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.ops.object.delete()
    return renders_count

def main():
    """Main function to process chair models and render them."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="src/data/raw")
    parser.add_argument("--output_dir", default="src/data/renders")
    parser.add_argument("--angles", type=int, default=12)
    parser.add_argument("--elevations", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--material_variations", type=int, default=3)
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_scene()
    setup_render_settings(resolution=args.resolution, samples=args.samples)

    chair_dirs = []
    for entry in os.listdir(args.models_dir):
        dir_path = os.path.join(args.models_dir, entry)
        if os.path.isdir(dir_path) and entry.startswith('chair-'):
            chair_dirs.append(dir_path)
    if not chair_dirs:
        print(f"No chair directories found in {args.models_dir}")
        return

    total_renders = 0
    for chair_dir in chair_dirs:
        chair_name = os.path.basename(chair_dir)
        print(f"Processing {chair_name}...")

        source_dir = os.path.join(chair_dir, 'source')
        model_files = []
        for ext in ['.obj', '.fbx', '.blend', '.gltf', '.glb']:
            model_files.extend(glob.glob(os.path.join(source_dir, f'*{ext}')))
        if not model_files:
            print(f"No supported model files found in {source_dir}")
            continue

        model_file = model_files[0]
        print(f"Using model: {model_file}")

        # Check if a textures folder exists in the chair directory.
        textures_dir = os.path.join(chair_dir, 'textures')
        if not os.path.isdir(textures_dir):
            textures_dir = None

        try:
            chair, materials = import_chair_model(model_file)
            material_variations = create_material_variations(materials, textures_dir)
            chair_render_dir = os.path.join(args.output_dir, chair_name)
            os.makedirs(chair_render_dir, exist_ok=True)
            for var_idx in range(min(args.material_variations, len(material_variations))):
                apply_material(chair, material_variations[var_idx])
                var_dir = os.path.join(chair_render_dir, f"var_{var_idx:02d}")
                os.makedirs(var_dir, exist_ok=True)
                num_renders = render_chair_angles(chair, var_dir, num_angles=args.angles, elevation_angles=args.elevations)
                total_renders += num_renders
                print(f"Completed {num_renders} renders for {chair_name} variation {var_idx+1}/{args.material_variations}")
        except Exception as e:
            print(f"Error processing {chair_name}: {str(e)}")
            continue

    print(f"Rendering complete. Total renders: {total_renders}")

if __name__ == "__main__":
    main()
