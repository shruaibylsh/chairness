import bpy
import math
import mathutils
import os

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def create_camera():
    bpy.ops.object.camera_add(location=(0, -3, 2))
    cam = bpy.context.active_object
    # Point camera at the origin (we’ll recenter models to (0,0,0))
    cam.rotation_euler = (math.radians(60), 0, 0)
    bpy.context.scene.camera = cam
    return cam

def set_render_settings():
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

def recenter_object(obj):
    # Compute the bounding box center in object space
    bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(bbox, mathutils.Vector()) / 8.0
    # Shift object so that its center is at its origin
    obj.location -= bbox_center
    # Reset origin to geometry center based on bounds
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

def import_model(source_dir):
    for f in os.listdir(source_dir):
        if f.lower().endswith(".obj") or f.lower().endswith(".fbx"):
            return os.path.join(source_dir, f)
    return None

def import_file(file_path):
    if file_path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=file_path, forward_axis='Y', up_axis='Z')
    elif file_path.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=file_path)
    else:
        raise ValueError("Unsupported file type")

def prepare_model_group(imported_objs):
    """
    If multiple objects are imported, create an empty pivot and parent them to it.
    Otherwise, return the single object.
    """
    if len(imported_objs) == 1:
        return imported_objs[0]
    else:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        pivot = bpy.context.active_object
        for obj in imported_objs:
            obj.select_set(True)
        pivot.select_set(True)
        bpy.context.view_layer.objects.active = pivot
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
        return pivot

def get_texture_file(textures_dir, keyword):
    """Return the first file in textures_dir containing keyword (case-insensitive)"""
    for fname in os.listdir(textures_dir):
        if keyword.lower() in fname.lower():
            return os.path.join(textures_dir, fname)
    return None

def apply_textures(model_dir, group_obj, model_name): # TODO: Make this more robust
    """
    Look for a textures folder and, if found, apply an albedo texture
    to a new material that is assigned to all mesh objects in group_obj.
    """
    textures_dir = os.path.join(model_dir, "textures")
    if not os.path.isdir(textures_dir):
        print(f"No textures folder found in {model_dir}")
        return

    # Try common naming conventions for the albedo texture:
    albedo_path = get_texture_file(textures_dir, "albedo")
    if not albedo_path:
        print(f"No albedo texture found in {textures_dir}")
        return

    # Create a new material with nodes enabled.
    mat = bpy.data.materials.new(name=f"{model_name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes.
    for node in nodes:
        nodes.remove(node)

    # Create an Output node.
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    # Create a Principled BSDF node.
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Create an Image Texture node and load the albedo image.
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-300, 0)
    try:
        img = bpy.data.images.load(albedo_path)
    except Exception as e:
        print(f"Error loading image {albedo_path}: {e}")
        return
    tex_node.image = img

    # Connect the texture color to the BSDF's Base Color.
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # Assign the new material to all mesh objects.
    # If group_obj is an EMPTY (the pivot), assign material to all its mesh children.
    if group_obj.type == 'EMPTY':
        for child in group_obj.children:
            if child.type == 'MESH':
                if child.data.materials:
                    child.data.materials[0] = mat
                else:
                    child.data.materials.append(mat)
    elif group_obj.type == 'MESH':
        if group_obj.data.materials:
            group_obj.data.materials[0] = mat
        else:
            group_obj.data.materials.append(mat)
    print(f"Applied texture from {albedo_path} to {model_name}")

def render_model(model_dir, model_name):
    clear_scene()
    create_camera()
    set_render_settings()

    # Build path to model source
    source_dir = os.path.join(model_dir, "source")
    model_file = import_model(source_dir)
    if not model_file:
        print(f"No valid model file found in {source_dir} for {model_name}")
        return

    print(f"Importing model file: {model_file}")
    try:
        import_file(model_file)
    except Exception as e:
        print(f"Error importing {model_file}: {e}")
        return

    # Get imported objects (mesh and empty objects)
    imported_objs = [obj for obj in bpy.context.selected_objects if obj.type in {'MESH', 'EMPTY'}]
    if not imported_objs:
        print(f"No objects imported for {model_name}")
        return

    # Recenter each mesh object.
    for obj in imported_objs:
        if obj.type == 'MESH':
            recenter_object(obj)

    # Group objects for unified rotation.
    group_obj = prepare_model_group(imported_objs)

    # Apply textures from the textures folder.
    apply_textures(model_dir, group_obj, model_name)

    # Save initial rotation.
    initial_rot = group_obj.rotation_euler.copy()

    # Prepare output directory.
    base_dir = os.path.dirname(os.path.dirname(bpy.data.filepath))
    output_dir = os.path.join(base_dir, "renders", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Render 12 views (rotating the group pivot by 30° increments).
    for i in range(12):
        angle = math.radians(i * 30)
        group_obj.rotation_euler = (initial_rot.x,
                                    initial_rot.y,
                                    initial_rot.z + angle)
        bpy.context.view_layer.update()
        
        output_filepath = os.path.join(output_dir, f"{model_name}_angle_{i:02d}.png")
        bpy.context.scene.render.filepath = output_filepath
        print(f"Rendering {model_name} at angle {math.degrees(angle):.1f}° -> {output_filepath}")
        bpy.ops.render.render(write_still=True)

def render_all_models():
    blend_file_path = bpy.data.filepath
    if not blend_file_path:
        print("Please save your .blend file first!")
        return
    base_dir = os.path.dirname(os.path.dirname(blend_file_path))
    models_dir = os.path.join(base_dir, "models")
    for d in os.listdir(models_dir):
        model_path = os.path.join(models_dir, d)
        if os.path.isdir(model_path) and d.startswith("chair-"):
            print(f"\n--- Processing model: {d} ---")
            render_model(model_path, d)

if __name__ == "__main__":
    render_all_models()
    print("\n=== All Models Rendered ===")
