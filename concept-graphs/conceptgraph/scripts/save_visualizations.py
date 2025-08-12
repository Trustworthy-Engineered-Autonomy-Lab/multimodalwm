import cv2
import os
import copy
import json
import pickle
import gzip
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip
import distinctipy

# Set Open3D to headless mode
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects

def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """Create a colored mesh sphere."""
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./visualizations", 
                        help="Directory to save visualization images")
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    parser.add_argument("--query", type=str, default=None,
                        help="Text query for CLIP similarity visualization")
    
    # To inspect the results of merge_overlap_objects
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

def load_result(result_path):
    """Load results from pickle file."""
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary!")
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
    
    return objects, bg_objects, class_colors

def save_visualization(geometries, output_path, width=1280, height=720):
    """Save a visualization image without interactive display."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set view parameters (you may need to adjust these)
    ctr = vis.get_view_control()
    # ctr.set_front([0, 0, -1])
    # ctr.set_lookat([0, 0, 0])
    # ctr.set_up([0, -1, 0])
    # ctr.set_zoom(0.8)
    
    vis.capture_screen_image(output_path)
    vis.destroy_window()

def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        
        if result_path is None:
            print("Only saving the pointcloud...")
            save_visualization([global_pcd], os.path.join(output_dir, "global_pcd.png"))
            return
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    # Initialize CLIP if needed
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    # Process background objects
    if bg_objects is not None:
        indices_bg = []
        for obj_idx, obj in enumerate(objects):
            if obj['is_background']:
                indices_bg.append(obj_idx)
    
    # Sub-sample point clouds
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get object classes
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        obj_classes = np.asarray(obj['class_id'])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Create different visualizations
    print("Creating visualizations...")
    
    # 1. Color by class
    print("- Class-colored visualization")
    for i in range(len(objects)):
        pcd = pcds[i]
        obj_class = object_classes[i]
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(class_colors[str(obj_class)], (len(pcd.points), 1))
        )
    save_visualization(pcds + bboxes, os.path.join(output_dir, "class_colored.png"))
    
    # 2. Color by RGB
    print("- RGB-colored visualization")
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd.colors = objects[i]['pcd'].colors
    save_visualization(pcds + bboxes, os.path.join(output_dir, "rgb_colored.png"))
    
    # 3. Color by instance
    print("- Instance-colored visualization")
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(instance_colors[i, :3], (len(pcd.points), 1))
        )
    save_visualization(pcds + bboxes, os.path.join(output_dir, "instance_colored.png"))
    
    # 4. CLIP similarity if query provided
    if args.query and not args.no_clip:
        print(f"- CLIP similarity visualization for query: '{args.query}'")
        text_queries = [args.query]
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        max_prob_object = objects[max_prob_idx]
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"Location xyz: {max_prob_object['bbox'].center}")
        
        for i in range(len(objects)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile([
                    similarity_colors[i, 0].item(),
                    similarity_colors[i, 1].item(),
                    similarity_colors[i, 2].item()
                ], (len(pcd.points), 1))
            )
        
        save_visualization(pcds + bboxes, os.path.join(output_dir, f"clip_similarity_{args.query.replace(' ', '_')}.png"))
    
    # 5. Scene graph if edge file provided
    if args.edge_file is not None:
        print("- Scene graph visualization")
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        
        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            radius = 0.10
            obj_centers.append(center)
            
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )
            scene_graph_geometries.extend(line_mesh.cylinder_segments)
        
        save_visualization(pcds + bboxes + scene_graph_geometries, 
                         os.path.join(output_dir, "scene_graph.png"))
    
    print(f"All visualizations saved to {output_dir}")
    
    # Save object information to text file
    info_file = os.path.join(output_dir, "object_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Total objects: {len(objects)}\n")
        f.write(f"Background objects: {len(bg_objects) if bg_objects else 0}\n\n")
        
        for i, obj in enumerate(objects):
            f.write(f"Object {i}:\n")
            f.write(f"  Class: {obj.get('class_name', 'Unknown')}\n")
            f.write(f"  Center: {obj['bbox'].center}\n")
            f.write(f"  Points: {len(obj['pcd'].points)}\n")
            f.write(f"  Is background: {obj.get('is_background', False)}\n\n")
    
    print(f"Object information saved to {info_file}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)