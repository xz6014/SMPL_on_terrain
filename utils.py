import scipy.ndimage.filters as filters
from dataset.data_processing.Learning import RBF
from lib.quaternion import *
from lib.meshviewer import Mesh
from os.path import exists
import trimesh
from copy import deepcopy

SCALE = 1 / 16.589719478653578 / 5.644
H_SCALE, V_SCALE = 3.937007874 * SCALE, 100 * SCALE


def cat_zero(arr):
    arr = arr.reshape(-1, 2)
    return np.stack([arr[:, 0], np.zeros_like(arr[:, 0]), arr[:, 1]], -1)


def build_watertight_mesh(heightmap, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset=0., base_height=-1,
                          simplification_factor=0.5):
    """
    Build a simplified, watertight trimesh.Mesh from a heightmap.

    Parameters:
    heightmap (numpy.ndarray): 2D array of height values
    scale_x (float): Scale factor for x-axis
    scale_y (float): Scale factor for y-axis
    scale_z (float): Scale factor for z-axis (height)
    offset (float): Offset for x and y coordinates
    base_height (float): Height of the base of the mesh
    simplification_factor (float): Proportion of faces to keep (0.0 to 1.0)

    Returns:
    trimesh.Mesh: A simplified, watertight mesh representation of the heightmap
    """
    m, n = heightmap.shape

    # Create vertex array for the top surface
    y, x = np.meshgrid(np.arange(m), np.arange(n))
    x = x - offset
    y = y - offset
    z = heightmap * scale_z
    top_vertices = np.stack([x.flatten() * scale_x,
                             z.flatten(),
                             y.flatten() * scale_y], axis=1)

    # Create vertex array for the bottom surface
    bottom_vertices = np.copy(top_vertices)
    bottom_vertices[:, 1] = base_height

    # Combine top and bottom vertices
    vertices = np.vstack([top_vertices, bottom_vertices])

    # Create face array for the top surface
    top_faces = []
    for i in range(m - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = (i + 1) * n + j
            v3 = v2 + 1
            top_faces.extend([[v0, v2, v1], [v1, v2, v3]])

    # Create face array for the bottom surface
    bottom_faces = []
    offset = m * n
    for i in range(m - 1):
        for j in range(n - 1):
            v0 = offset + i * n + j
            v1 = v0 + 1
            v2 = offset + (i + 1) * n + j
            v3 = v2 + 1
            bottom_faces.extend([[v0, v1, v2], [v1, v3, v2]])

    # Create face array for the sides
    side_faces = []
    # Front side
    for i in range(n - 1):
        v0, v1 = i, i + 1
        v2, v3 = offset + i, offset + i + 1
        side_faces.extend([[v0, v2, v1], [v1, v2, v3]])
    # Back side
    for i in range(n - 1):
        v0, v1 = (m - 1) * n + i, (m - 1) * n + i + 1
        v2, v3 = offset + v0, offset + v0 + 1
        side_faces.extend([[v0, v1, v2], [v1, v3, v2]])
    # Left side
    for i in range(m - 1):
        v0, v1 = i * n, (i + 1) * n
        v2, v3 = offset + v0, offset + v1
        side_faces.extend([[v0, v2, v1], [v1, v2, v3]])
    # Right side
    for i in range(m - 1):
        v0, v1 = (i + 1) * n - 1, (i + 2) * n - 1
        v2, v3 = offset + v0, offset + v1
        side_faces.extend([[v0, v1, v2], [v1, v3, v2]])

    # Combine all faces
    faces = np.array(top_faces + bottom_faces + side_faces)

    # Create the initial mesh
    initial_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Calculate target number of faces
    target_faces = int(len(initial_mesh.faces) * simplification_factor)
    # # Simplify the mesh
    simplified_mesh = initial_mesh.simplify_quadric_decimation(target_faces)
    # Ensure the mesh is watertight
    simplified_mesh.fill_holes()
    simplified_mesh.remove_degenerate_faces()
    simplified_mesh.remove_duplicate_faces()

    # Convert back to our Mesh class
    final_mesh = Mesh(vertices=simplified_mesh.vertices, faces=simplified_mesh.faces)

    return final_mesh


def patchfunc(P, Xp, hscale=H_SCALE, vscale=V_SCALE):
    Xp = Xp / hscale + np.array([P.shape[1] // 2, P.shape[2] // 2])

    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1] - 1, P.shape[2] - 1]))
    X1 = np.clip(np.ceil(Xp).astype(np.int), 0, np.array([P.shape[1] - 1, P.shape[2] - 1]))

    H0 = P[:, X0[:, 0], X0[:, 1]]
    H1 = P[:, X0[:, 0], X1[:, 1]]
    H2 = P[:, X1[:, 0], X0[:, 1]]
    H3 = P[:, X1[:, 0], X1[:, 1]]

    HL = (1 - A[:, 0]) * H0 + (A[:, 0]) * H2
    HR = (1 - A[:, 0]) * H1 + (A[:, 0]) * H3

    return (vscale * ((1 - A[:, 1]) * HL + (A[:, 1]) * HR))[..., np.newaxis]


def process_heights(global_positions, nsamples=3, name='flat'):

    rng = np.random.RandomState(1234)

    # """ Load Terrain Patches """
    if exists('/BS/XZ_project6/work/PFNN'):
        patches_database = np.load('/BS/XZ_project6/work/PFNN/patches.npz')
    else:
        patches_database = np.load('../PFNN/patches.npz')

    patches = patches_database['X'].astype(np.float32) * SCALE

    """ Do FK """
    if 'LocomotionFlat12_000' in name:
        type = 'jumpy'
    elif 'NewCaptures01_000' in name:
        type = 'flat'
    elif 'NewCaptures02_000' in name:
        type = 'flat'
    elif 'NewCaptures03_000' in name:
        type = 'jumpy'
    elif 'NewCaptures03_001' in name:
        type = 'jumpy'
    elif 'NewCaptures03_002' in name:
        type = 'jumpy'
    elif 'NewCaptures04_000' in name:
        type = 'jumpy'
    elif 'WalkingUpSteps06_000' in name:
        type = 'beam'
    elif 'WalkingUpSteps09_000' in name:
        type = 'flat'
    elif 'WalkingUpSteps10_000' in name:
        type = 'flat'
    elif 'WalkingUpSteps11_000' in name:
        type = 'flat'
    elif 'Flat' in name:
        type = 'flat'
    # elif len(name.split('_')) == 2:
    #     type = 'flat'
    else:
        type = 'rocky'

    """ Extract Forward Direction """
    SDR_L, SDR_R, HIP_L, HIP_R = 13, 14, 1, 2
    across = (
            (global_positions[:, SDR_L] - global_positions[:, SDR_R]) +
            (global_positions[:, HIP_L] - global_positions[:, HIP_R]))
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    """ Smooth Forward Direction """

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]]).repeat(forward.shape[0], 0)

    """ Foot Contacts """
    FOOT_L, FOOT_R = [7, 10], [8, 11]
    fid_l, fid_r = np.array(FOOT_L), np.array(FOOT_R)
    velfactor = np.array([0.02, 0.02]) * SCALE

    feet_l_x = (global_positions[1:, fid_l, 0] - global_positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (global_positions[1:, fid_l, 1] - global_positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (global_positions[1:, fid_l, 2] - global_positions[:-1, fid_l, 2]) ** 2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))

    feet_r_x = (global_positions[1:, fid_r, 0] - global_positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (global_positions[1:, fid_r, 1] - global_positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (global_positions[1:, fid_r, 2] - global_positions[:-1, fid_r, 2]) ** 2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))

    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)

    """ Toe and Heel Heights """

    toe_h, heel_h = 4.0 * SCALE, 5.0 * SCALE

    """ Foot Down Positions """

    feet_down = np.concatenate([
        global_positions[feet_l[:, 0], fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_l[:, 1], fid_l[1]] - np.array([0, toe_h, 0]),
        global_positions[feet_r[:, 0], fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_r[:, 1], fid_r[1]] - np.array([0, toe_h, 0])
    ], axis=0)

    """ Foot Up Positions """

    feet_up = np.concatenate([
        global_positions[~feet_l[:, 0], fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:, 1], fid_l[1]] - np.array([0, toe_h, 0]),
        global_positions[~feet_r[:, 0], fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:, 1], fid_r[1]] - np.array([0, toe_h, 0])
    ], axis=0)

    """ Down Locations """
    feet_down_xz = np.concatenate([feet_down[:, 0:1], feet_down[:, 2:3]], axis=-1)
    feet_down_xz_mean = feet_down_xz.mean(axis=0)
    feet_down_y = feet_down[:, 1:2]
    feet_down_y_mean = feet_down_y.mean(axis=0)
    """ Up Locations """

    feet_up_xz = np.concatenate([feet_up[:, 0:1], feet_up[:, 2:3]], axis=-1)
    feet_up_y = feet_up[:, 1:2]

    if len(feet_down_xz) == 0:

        """ No Contacts """
        terrains = [None] * nsamples

    elif type == 'flat':

        """ Flat """
        terrains = [None] * nsamples
    else:

        """ Terrain Heights """

        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)

        """ Fitting Error """

        terr_down_err = 0.1 * ((
                                       (terr_down_y - terr_down_y_mean[:, np.newaxis]) -
                                       (feet_down_y - feet_down_y_mean)[np.newaxis]) ** 2)[..., 0].mean(axis=1)

        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:, np.newaxis]) -
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0) ** 2)[..., 0].mean(axis=1)

        """ Jumping Error """

        if type == 'jumpy':
            terr_over_minh = 5.0 * SCALE
            terr_over_err = (np.maximum(
                ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                (terr_up_y - terr_down_y_mean[:, np.newaxis]), 0.0) ** 2)[..., 0].mean(axis=1)
        else:
            terr_over_err = 0.0

        """ Fitting Terrain to Walking on Beam """

        if type == 'beam':

            beam_samples = 1
            beam_min_height = 40.0 * SCALE

            beam_c = global_positions[:, 0]
            beam_c_xz = np.concatenate([beam_c[:, 0:1], beam_c[:, 2:3]], axis=-1)
            beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

            beam_o = (
                    beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) *
                    rng.normal(size=(len(beam_c) * beam_samples, 3)) * SCALE)

            beam_o_xz = np.concatenate([beam_o[:, 0:1], beam_o[:, 2:3]], axis=-1)
            beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

            beam_pdist = np.sqrt(((beam_o[:, np.newaxis] - beam_c[np.newaxis, :]) ** 2).sum(axis=-1))
            beam_far = (beam_pdist > 15 * SCALE).all(axis=1)

            terr_beam_err = (np.maximum(beam_o_y[:, beam_far] -
                                        (beam_c_y.repeat(beam_samples, axis=1)[:, beam_far] -
                                         beam_min_height), 0.0) ** 2)[..., 0].mean(axis=1)

        else:
            terr_beam_err = 0.0

        """ Final Fitting Error """

        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err

        """ Best Fitting Terrains """

        terr_ids = np.argsort(terr)[:nsamples]
        terr_patches = patches[terr_ids]

        # Visualize the skeleton and terrain
        terr_basic_func = lambda Xp: (
                (patchfunc(terr_patches, Xp - feet_down_xz_mean) -
                 terr_down_y_mean[terr_ids][:, np.newaxis]) + feet_down_y_mean)

        """ Terrain Fit Editing """
        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        terr_fine_func = [RBF(smooth=0.1, function='linear', epsilon=1e-10) for _ in range(nsamples)]
        for i in range(nsamples): terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
        # visualise
        h_scale, v_scale = H_SCALE, V_SCALE
        terrains = []
        for i in range(nsamples):
            scale_orig = terr_patches[0].shape[0]
            best_terrain_mesh = build_watertight_mesh(terr_patches[i], scale_x=h_scale, scale_y=h_scale,
                                                      scale_z=v_scale,
                                                      offset=scale_orig // 2, base_height=-1, simplification_factor=0.1)
            best_terrain_mesh.fix_normals()
            best_terrain_mesh.vertices[:, 1] = deepcopy(best_terrain_mesh.vertices)[:, 1] - terr_down_y_mean[
                terr_ids[i]] + feet_down_y_mean
            best_terrain_mesh.vertices[:, [0, 2]] += feet_down_xz_mean
            terrains.append(best_terrain_mesh)
    return terrains

