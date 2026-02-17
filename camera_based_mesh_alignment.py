import pandas as pd
import os
import json
import numpy as np


def least_squares_fit(A, B):
    """
    Compute the least-squares linear transformation that maps point set B to point set A.

    Parameters
    ----------
    A : np.ndarray of shape (n, 3)
        Target 3D point set.
    B : np.ndarray of shape (n, 3)
        Source 3D point set.

    Returns
    -------
    M : np.ndarray of shape (3, 3)
        Linear transformation matrix.
    t : np.ndarray of shape (3,)
        Translation vector.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    assert A.shape == B.shape, "Input point sets must have identical dimensions."

    n, d = A.shape  # n = number of points, d = dimensionality (expected 3)

    # Step 1: Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Step 2: Center point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Step 3: Compute covariance matrices
    Q = np.dot(A_centered.T, B_centered) / n
    P = np.dot(B_centered.T, B_centered) / n

    # Step 4: Solve for linear transformation matrix
    P_inv = np.linalg.inv(P)
    M = np.dot(Q, P_inv)

    # Step 5: Compute translation vector
    t = centroid_A - np.dot(M, centroid_B)

    return M, t


def apply_transformation(points, M, t):
    """
    Apply a linear transformation and translation to a 3D point set.

    Parameters
    ----------
    points : np.ndarray of shape (n, 3) or (3,)
        Input 3D points.
    M : np.ndarray of shape (3, 3)
        Transformation matrix.
    t : np.ndarray of shape (3,)
        Translation vector.

    Returns
    -------
    np.ndarray
        Transformed 3D points.
    """
    return np.dot(points, M.T) + t


def modify_vertices(obj_file, output_file, R, t):
    """
    Read an OBJ file, transform its vertices, and write the result to a new file.

    Parameters
    ----------
    obj_file : str
        Path to the input OBJ file.
    output_file : str
        Path to the output OBJ file.
    R : np.ndarray (3x3)
        Linear transformation matrix.
    t : np.ndarray (3,)
        Translation vector.
    """

    # Additional rotation applied to correct 90-degree orientation mismatch (sometimes other 90-degree rotation matrix is needed)
    rotation = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    with open(obj_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            # Process only vertex definitions (lines starting with 'v ')
            if line.startswith('v '):
                tokens = line.split()
                vertex = np.array(tokens[1:4], dtype=float)

                # Apply transformation and corrective rotation
                transformed = apply_transformation(vertex, R, t)
                transformed = transformed @ rotation

                new_vertex_line = f"v {transformed[0]} {transformed[1]} {transformed[2]}\n"
                file.write(new_vertex_line)
            else:
                file.write(line)


# Configuration parameters
capture_sets = ["50", "125", "250", "500"]
mesh_partitions = [1, 4, 16, 32]
building_name = "schwarzenbach-houses-with-interior2"
texturizers = [
    "Texturizer5000k",
    "Texturizer12500k",
    "Texturizer25000k",
    "Texturizer50000k",
    "Texturizer100000k"
]

# -------------------------------------------------------------------------
# Main Processing Loop
# Iterates over capture subsets, texturizers, and mesh partitions
# -------------------------------------------------------------------------

for capture_subset in capture_sets:
    for texturizer_name in texturizers:
        for mesh_partition in mesh_partitions:

            # -------------------------------------------------------------
            # 1. Define input dataset paths
            # -------------------------------------------------------------

            original_dataset_path = (
                f"/Datasets/{building_name}/ima_1000")

            # Select the appropriate capture subset
            if capture_subset == "1000":
                working_dataset_path = original_dataset_path
            else:
                working_dataset_path = (
                    f"/Datasets/{building_name}/subset_{capture_subset}"
                )

            # -------------------------------------------------------------
            # 2. Define reconstruction output paths
            # -------------------------------------------------------------

            reconstruction_root = (
                f"/ReconstructedModels/{building_name}/NEW_{building_name}_{capture_subset}_sfm_NORMAL_INCREMENTAL_g1/{texturizer_name}/{mesh_partition}"
            )

            obj_input_path = os.path.join(reconstruction_root, "textured_mesh.obj")
            obj_output_path = os.path.join(
                reconstruction_root,
                "textured_mesh_final_modified.obj"
            )

            camera_calibration_path = os.path.join(reconstruction_root, "CameraCalib.txt")
            picture_names_path = os.path.join(reconstruction_root, "PictureNames.txt")

            # -------------------------------------------------------------
            # 3. Load capture list
            # -------------------------------------------------------------

            capture_files = sorted(os.listdir(working_dataset_path))

            # -------------------------------------------------------------
            # 4. Load reconstruction camera parameters
            # -------------------------------------------------------------

            with open(camera_calibration_path, 'r') as f:
                calibration_lines = f.readlines()[1:]  # First line = number of images

            with open(picture_names_path, 'r') as f:
                reconstruction_images = f.readlines()

            # DataFrame initialization
            reconstruction_data = pd.DataFrame(columns=["K_re", "R_re", "C_re", "t_re"])
            reconstruction_names = pd.DataFrame(columns=["path_re"])

            # Parse camera calibration file
            for line in calibration_lines:
                values = line.split()

                K_re = np.array(values[0:9], dtype=float).reshape(3, 3)
                R_re = np.array(values[9:18], dtype=float).reshape(3, 3)
                C_re = np.array(values[18:21], dtype=float).reshape(3, 1)
                t_re = -(R_re @ C_re)

                reconstruction_data = pd.concat(
                    [
                        reconstruction_data,
                        pd.DataFrame([{
                            "K_re": K_re,
                            "R_re": R_re,
                            "C_re": C_re,
                            "t_re": t_re
                        }])
                    ],
                    ignore_index=True
                )

            # Parse image names
            for image_path in reconstruction_images:
                image_name = image_path.split("/")[-1].strip()[:-4]

                reconstruction_names = pd.concat(
                    [
                        reconstruction_names,
                        pd.DataFrame([{"path_re": image_name}])
                    ],
                    ignore_index=True
                )

            # Merge names and parameters
            reconstruction_df = pd.concat(
                [reconstruction_names, reconstruction_data],
                axis=1
            )

            # -------------------------------------------------------------
            # 5. Compute camera center correspondences
            # -------------------------------------------------------------

            original_centers = []
            reconstructed_centers = []

            for image_file in capture_files:

                # Skip auxiliary files
                if image_file.startswith("images_info"):
                    continue

                json_filename = image_file.replace(".png", ".txt")
                json_path = os.path.join(original_dataset_path, json_filename)

                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        metadata = json.loads(f.read())
                    except json.JSONDecodeError:
                        continue

                image_id = json_filename.split(".")[0]

                # Retrieve reconstructed camera center
                C_reconstructed = reconstruction_df.loc[
                    reconstruction_df["path_re"] == image_id,
                    "C_re"
                ].iloc[0]

                # Compute original camera center
                R_original = np.array(metadata["R"], dtype=float).reshape(3, 3)
                t_original = np.array(metadata["t"], dtype=float).reshape(3, 1)
                C_original = -(np.linalg.inv(R_original) @ t_original)

                original_centers.append(C_original)
                reconstructed_centers.append(C_reconstructed)

            # -------------------------------------------------------------
            # 6. Compute optimal alignment transformation
            # -------------------------------------------------------------

            original_array = np.vstack([c.flatten() for c in original_centers])
            reconstructed_array = np.vstack([c.flatten() for c in reconstructed_centers])

            R_opt, t_opt = least_squares_fit(original_array, reconstructed_array)

            # -------------------------------------------------------------
            # 7. Apply transformation to mesh
            # -------------------------------------------------------------

            modify_vertices(obj_input_path, obj_output_path, R_opt, t_opt)

            print(
                f"[OK] Processed: "
                f"subset={capture_subset}, "
                f"texturizer={texturizer_name}, "
                f"partition={mesh_partition}"
            )