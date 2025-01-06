import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

def straighten_mesh_improved(input_path, output_path):
    """
    Straighten a 3D mesh by analyzing and correcting its forward tilt.
    
    Parameters:
    input_path: str, path to input .obj file
    output_path: str, path where to save straightened mesh
    """
    # Load the mesh
    mesh = trimesh.load(input_path)
    vertices = mesh.vertices
    
    # Calculate the bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    
    # Find the forward direction (assuming Z is forward)
    forward_points = vertices[vertices[:, 2] > np.mean(vertices[:, 2])]
    if len(forward_points) == 0:
        forward_points = vertices
    
    # Calculate the average tilt
    forward_center = np.mean(forward_points, axis=0)
    back_points = vertices[vertices[:, 2] < np.mean(vertices[:, 2])]
    back_center = np.mean(back_points, axis=0)
    
    # Calculate tilt angle in XZ plane
    direction = forward_center - back_center
    angle_xz = np.arctan2(direction[1], direction[2])  # Y-up rotation
    
    # Create rotation matrix for X-axis to correct forward tilt
    rot_x = Rotation.from_euler('x', -angle_xz, degrees=False)
    rotation_matrix = rot_x.as_matrix()
    
    # Apply rotation
    mesh.vertices = np.dot(vertices - mesh.center_mass, rotation_matrix) + mesh.center_mass
    
    # Optional: Ensure model is perfectly upright by aligning to Y-axis
    bbox = mesh.bounds
    height_axis = np.argmax(np.ptp(bbox, axis=0))
    if height_axis != 1:  # If height axis is not Y
        # Rotate to align height with Y-axis
        rot_align = np.eye(3)
        rot_align[1, 1] = 0
        rot_align[height_axis, height_axis] = 0
        rot_align[1, height_axis] = 1
        rot_align[height_axis, 1] = -1
        mesh.vertices = np.dot(mesh.vertices - mesh.center_mass, rot_align) + mesh.center_mass
    
    # Save the straightened mesh
    mesh.export(output_path)
    
    return mesh

# Additional utility function to try multiple angles
def try_multiple_angles(input_path, output_path, angle_range=(-30, 30), steps=60):
    """
    Try multiple rotation angles and save all versions.
    
    Parameters:
    input_path: str, path to input .obj file
    output_path: str, base path for output files (will append angle)
    angle_range: tuple, (min_angle, max_angle) in degrees
    steps: int, number of angles to try
    """
    mesh = trimesh.load(input_path)
    base_name = output_path.replace('.obj', '')
    
    angles = np.linspace(angle_range[0], angle_range[1], steps)
    for angle in angles:
        # Create rotation matrix
        rot = Rotation.from_euler('x', np.radians(angle))
        rot_matrix = rot.as_matrix()
        
        # Apply rotation
        rotated_vertices = np.dot(mesh.vertices - mesh.center_mass, rot_matrix) + mesh.center_mass
        new_mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=mesh.faces)
        
        # Save with angle in filename
        output_name = f"{base_name}_angle_{angle:.1f}.obj"
        new_mesh.export(output_name)

# Example usage
if __name__ == "__main__":
    input_file = "8fz19h2h_0.obj"
    output_file = "straightened_model.obj"
    
    # Try the main straightening function
    straighten_mesh_improved(input_file, output_file)
    
    # Or try multiple angles to find the best one
    # try_multiple_angles(input_file, output_file)