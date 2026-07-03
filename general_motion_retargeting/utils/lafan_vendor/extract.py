import re, os, ntpath
import numpy as np
from . import utils

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split()
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            # data_block *= 0
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0]:
                    q, x = utils.quat_fk(anim.quats[i: i+window], anim.pos[i: i+window], anim.parents)
                    # Extract contacts
                    c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r


def get_train_stats(bvh_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print('Building the train set...')
    xtrain, qtrain, parents, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)

    return x_mean, x_std, offsets


def read_bvh_leju(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split('    ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)

#read qmai bvh
def qmai_read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    channelmap = {
        'Xrotation': 'x',
        'Yrotation': 'y',
        'Zrotation': 'z'
    }

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split()
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            # data_block *= 0
            N = len(parents)
            fi = i - start if start else i

            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    #将第一帧替换为标准Apose，并在第一帧和第二帧之间做插值
    from scipy.spatial.transform import Rotation as R
    #标准Apose数据(gmr)
    Apose = np.array([-0.000300, 86.498000, 0.007900, -0.248805, 1.517919, 0.000000, 
             -0.421814, -0.014076, 5.781131, -0.241866, 0.011152, -2.725965, 
             -0.276076, 0.236178, -13.389382, -2.519453, 1.528804, 28.332168, 
             3.811490, -0.589678, -25.466202, 0.000000, 0.000000, 0.000000, 
             -26.869677, -1.711120, -0.040722, -51.925555, 14.512137, 33.183348, 
             -0.000000, -11.923160, 0.000000, 0.080479, -0.005239, -0.988820, 
             0.000000, 0.000000, 0.607176, 0.005215, 0.000010, -0.110703, 
             -0.012689, 0.000454, -0.146434, 0.022390, -0.001199, 4.934506, 
             0.000000, 0.000000, 0.746397, 0.000014, 0.006565, 0.122572, 
             -0.000022, -0.014325, 0.164521, 2.447046, -0.079831, 2.056764, 
             0.000000, 0.000000, 0.740482, -0.000002, 0.007173, -0.014980, 
             -0.000004, -0.015017, 0.027776, 4.260386, -0.094452, 1.375350, 
             0.000000, 0.000000, 0.754839, 0.000000, 0.009448, 0.000229, 
             -0.000006, -0.022452, 0.026332, -6.917769, 0.291382, 2.291170, 
             0.000000, 0.000000, 0.754038, -0.000001, 0.004175, -0.008146, 
             -0.000001, -0.009275, 0.012467, 0.441624, -0.014543, 2.538997, 
             28.441529, 3.047523, -0.040131, 54.103908, -16.200624, 15.342980, 
             -0.000000, 11.419530, -0.000000, -0.080308, 0.004629, -0.724563, 
             0.000000, 0.000000, 0.472622, 0.000000, 0.000000, 0.008252, 
             0.000000, 0.000000, -0.001366, -7.969741, -5.107087, -8.050663, 
             0.000000, 0.000000, 0.467721, 0.000000, 0.000000, -0.015015, 
             0.000000, 0.000000, 0.025121, 6.271931, -1.933901, -4.817211, 
             0.000000, 0.000000, 0.481966, 0.000000, 0.000000, -0.004083, 
             0.000000, 0.000000, -0.000628, -1.013052, -2.505694, -6.799366, 
             0.000000, 0.000000, 0.479930, 0.000000, 0.000000, 0.003389, 
             0.000000, 0.000000, -0.007978, -3.109834, -2.403556, -3.139792, 
             0.000035, 0.004242, 0.480870, 0.000001, -0.004242, 0.042897, 
             0.000000, 0.000000, 0.013196, -4.610124, -2.491975, -2.359596, 
             5.234722, 24.690708, -17.805216, 0.000000, 0.000000, 30.673946, 
             -0.785893, -14.385931, -11.349765, 0.000000, 0.000000, 0.000000, 
             -5.586639, -26.735999, -15.184888, 0.000000, 0.000000, 30.388747, 
             1.552447, 8.335392, -12.940579, 0.000000, 0.000000, -0.000003])

    #备用，上面的Apose在soma中手会自穿，此版可用于soma
    # Apose = np.array([-0.000300, 86.498000, 0.007900, -0.248805, 1.517919, 0.000000, 
    #          -0.421814, -0.014076, 5.781131, -0.241866, 0.011152, -2.725965, 
    #          -0.276076, 0.236178, -13.389382, -2.519453, 1.528804, 28.332168, 
    #          3.811490, -0.589678, -25.466202, 0.000000, 0.000000, 0.000000, 
    #          -26.869677, -1.711120, -0.040722, -43.925555, 14.512137, 3.183348, 
    #          -0.000000, -25.000000, 0.000000, 0.080479, -0.005239, -0.988820, 
    #          0.000000, 0.000000, 0.607176, 0.005215, 0.000010, -0.110703, 
    #          -0.012689, 0.000454, -0.146434, 0.022390, -0.001199, 4.934506, 
    #          0.000000, 0.000000, 0.746397, 0.000014, 0.006565, 0.122572, 
    #          -0.000022, -0.014325, 0.164521, 2.447046, -0.079831, 2.056764, 
    #          0.000000, 0.000000, 0.740482, -0.000002, 0.007173, -0.014980, 
    #          -0.000004, -0.015017, 0.027776, 4.260386, -0.094452, 1.375350, 
    #          0.000000, 0.000000, 0.754839, 0.000000, 0.009448, 0.000229, 
    #          -0.000006, -0.022452, 0.026332, -6.917769, 0.291382, 2.291170, 
    #          0.000000, 0.000000, 0.754038, -0.000001, 0.004175, -0.008146, 
    #          -0.000001, -0.009275, 0.012467, 0.441624, -0.014543, 2.538997, 
    #         28.441529, 3.047523, -0.040131, 46.103908, -16.200624, -15.342980, 
    #          -0.000000, 25.000000, -0.000000, -0.080308, 0.004629, -0.724563, 
    #          0.000000, 0.000000, 0.472622, 0.000000, 0.000000, 0.008252, 
    #          0.000000, 0.000000, -0.001366, -7.969741, -5.107087, -8.050663, 
    #          0.000000, 0.000000, 0.467721, 0.000000, 0.000000, -0.015015, 
    #          0.000000, 0.000000, 0.025121, 6.271931, -1.933901, -4.817211, 
    #          0.000000, 0.000000, 0.481966, 0.000000, 0.000000, -0.004083, 
    #          0.000000, 0.000000, -0.000628, -1.013052, -2.505694, -6.799366, 
    #          0.000000, 0.000000, 0.479930, 0.000000, 0.000000, 0.003389, 
    #          0.000000, 0.000000, -0.007978, -3.109834, -2.403556, -3.139792, 
    #          0.000035, 0.004242, 0.480870, 0.000001, -0.004242, 0.042897, 
    #          0.000000, 0.000000, 0.013196, -4.610124, -2.491975, -2.359596, 
    #          5.234722, 24.690708, -17.805216, 0.000000, 0.000000, 30.673946, 
    #          -0.785893, -14.385931, -11.349765, 0.000000, 0.000000, 0.000000, 
    #          -5.586639, -26.735999, -15.184888, 0.000000, 0.000000, 30.388747, 
    #          1.552447, 8.335392, -12.940579, 0.000000, 0.000000, -0.000003])
    positions[0, 0:1] = Apose[0:3]
    rotations[0, :] = Apose[3:].reshape(len(parents), 3)
    
    if positions.shape[0] >= 2:
        pos0 = positions[0]  # 手动插入的起始帧
        pos1 = positions[1]  # 原始第一帧
        rot0 = rotations[0]
        rot1 = rotations[1]

        # 动态计算插入帧数
        pos_diff = np.abs(pos1 - pos0)
        rot_diff = np.abs(rot1 - rot0)
        max_pos_diff = np.max(pos_diff)
        max_rot_diff = np.max(rot_diff)

        insert_frames = 0
        pos_threshold = 3.0    # 位移超过这个值开始插值
        rot_threshold = 6.0  # 旋转角度超过这个值开始插值
        
        if max_pos_diff > pos_threshold or max_rot_diff > rot_threshold:
            # 按差异大小计算帧数（最小5，最大30）
            insert_frames = min(max(int(max_pos_diff/pos_threshold * 0.67), int(max_rot_diff/rot_threshold * 0.67)), 30)
            insert_frames = max(insert_frames, 2)
            print(f"A-Pose到动作第1帧间插入 {insert_frames} 帧平滑过渡")

        alphas = np.linspace(0, 1, insert_frames + 2)  # 包含首尾
        new_pos = []
        new_rot = []

        # 执行线性插值，生成过渡帧
        for a in alphas:
            # 位置：线性插值
            ipos = (1 - a) * pos0 + a * pos1

            # 旋转：**正确球面插值 SLERP**
            irot = np.zeros_like(rot0)
            for j in range(rot0.shape[0]):
                # 欧拉角 -> 四元数
                q0 = R.from_euler(order, rot0[j], degrees=True).as_quat()
                q1 = R.from_euler(order, rot1[j], degrees=True).as_quat()
                # 手动球面插值 Slerp
                dot = np.dot(q0, q1)
                if dot < 0.0:
                    q1 = -q1
                    dot = -dot
                if dot > 0.9995:
                    # 太近，直接线性插值
                    q = q0 * (1 - a) + q1 * a
                else:
                    theta = np.arccos(dot)
                    sin_theta = np.sin(theta)
                    q = (np.sin((1 - a) * theta) / sin_theta) * q0 + (np.sin(a * theta) / sin_theta) * q1

                # 四元数 -> 欧拉角
                irot[j] = R.from_quat(q).as_euler(order, degrees=True)

            new_pos.append(ipos)
            new_rot.append(irot)
        # 替换帧序列
        positions = np.concatenate([np.array(new_pos), positions[2:]], axis=0)
        rotations = np.concatenate([np.array(new_rot), rotations[2:]], axis=0)
    
    # 当前末尾帧
    end_pos = positions[-1, 0, :].copy()
    end_rot = rotations[-1].copy()

    # 肩到肘高于水平面（0度）时，该侧手臂经过胸前中间姿态。
    end_quat = utils.euler_to_quat(
        np.radians(end_rot)[np.newaxis, ...],
        order=order,
    )
    _, end_global_pos = utils.quat_fk(
        end_quat,
        positions[-1][np.newaxis, ...],
        parents,
    )
    end_global_pos = end_global_pos[0]
    raised_sides = []
    arm_raise_threshold = 0.0
    for side in ("Left", "Right"):
        shoulder_idx = names.index(f"{side}Shoulder")
        elbow_idx = names.index(f"{side}Elbow")
        upper_arm = (
            end_global_pos[elbow_idx]
            - end_global_pos[shoulder_idx]
        )
        upper_arm_elevation = np.degrees(np.arcsin(np.clip(
            upper_arm[1] / np.linalg.norm(upper_arm),
            -1.0,
            1.0,
        )))
        if upper_arm_elevation > arm_raise_threshold:
            raised_sides.append(side)
            print(
                f"{side}上臂抬升角 {upper_arm_elevation:.1f}°"
                f"超过阈值 {arm_raise_threshold:.1f}°"
            )

    if raised_sides:
        print(f"检测到高举手臂: {', '.join(raised_sides)}，该手臂先收至中间态再回A-Pose")
    else:
        print("未检测到高举手臂，直接回A-Pose")

    Apose_pos = Apose[0:3].copy()
    Apose_rot = Apose[3:].reshape(len(parents),3)
    # 中间态只保存 Collar/Shoulder（肩带/大臂）的 Z/Y/X 旋转。
    right_chest_guide = np.array([
        27.897471, 1.975414, -0.031463,
        40.696368, 45.749928, 8.138802,
    ]).reshape(2, 3)
    left_chest_guide = right_chest_guide * np.array([-1.0, -1.0, 1.0])
    chest_guide_values = {
        "Left": left_chest_guide,
        "Right": right_chest_guide,
    }
    arm_rotation_slices = {
        "Left": slice(
            names.index("LeftCollar"),
            names.index("LeftWrist") + 1,
        ),
        "Right": slice(
            names.index("RightCollar"),
            names.index("RightWrist") + 1,
        ),
    }

    first_target_pos = end_pos.copy()
    first_target_pos[1] = Apose_pos[1]
    first_target_rot = Apose_rot.copy()
    for side in raised_sides:
        arm_slice = arm_rotation_slices[side]
        # 肘和手腕保持动作末帧姿态，只替换肩带与大臂。
        first_target_rot[arm_slice] = end_rot[arm_slice]
        first_target_rot[arm_slice.start:arm_slice.start + 2] = (
            chest_guide_values[side]
        )
    raised_guide_indices = {
        joint_idx
        for side in raised_sides
        for joint_idx in (
            names.index(f"{side}Collar"),
            names.index(f"{side}Shoulder"),
        )
    }

    # 停顿帧
    pause_frames = 10
    pause_pos = np.repeat(positions[-1][np.newaxis,:,:], pause_frames, axis=0)
    pause_rot = np.repeat(rotations[-1][np.newaxis,:,:], pause_frames, axis=0)
    positions = np.concatenate([positions, pause_pos], axis=0)
    rotations = np.concatenate([rotations, pause_rot], axis=0)

    # 末尾 -> 高举侧：胸前中间姿态/低举侧：Apose
    diff_pos = np.abs(end_pos - first_target_pos)
    diff_rot = np.abs(end_rot - first_target_rot)
    frames1 = max(int(max(np.max(diff_pos)/3.0, np.max(diff_rot)/6.0)),30)
    first_target_name = "胸前中间姿态" if raised_sides else "A-Pose"
    print(f"动作最后一帧到{first_target_name}间插入 {frames1} 帧平滑过渡")

    alphas1 = np.linspace(0,1,frames1+2)
    new_pos1,new_rot1 = [],[]

    for a in alphas1:
        ipos = positions[-1].copy()
        ipos[0,0] = (1-a)*end_pos[0] + a*first_target_pos[0]
        ipos[0,1] = (1-a)*end_pos[1] + a*first_target_pos[1]
        ipos[0,2] = (1-a)*end_pos[2] + a*first_target_pos[2]
        new_pos1.append(ipos)

        irot = np.zeros_like(end_rot)
        for j in range(end_rot.shape[0]):
            if j in raised_guide_indices:
                # 肩带和大臂沿参考帧的欧拉通道插值。
                irot[j] = (
                    (1-a) * end_rot[j]
                    + a * first_target_rot[j]
                )
                continue

            q0 = R.from_euler(order,end_rot[j],degrees=True).as_quat()
            q1 = R.from_euler(order,first_target_rot[j],degrees=True).as_quat()
            dot = np.dot(q0,q1)
            if dot < 0:
                q1 = -q1
                dot = -dot
            # SLERP
            if dot>0.9995:
                q = q0*(1-a)+q1*a
            else:
                theta,sin_theta=np.arccos(dot),np.sin(np.arccos(dot))
                q = (np.sin((1-a)*theta)/sin_theta)*q0 + (np.sin(a*theta)/sin_theta)*q1
            irot[j] = R.from_quat(q).as_euler(order,degrees=True)
        new_rot1.append(irot)

    positions = np.concatenate([positions,np.array(new_pos1)],axis=0)
    rotations = np.concatenate([rotations,np.array(new_rot1)],axis=0)

    if raised_sides:
        # 高举侧：胸前中间姿态 -> Apose
        frames2 = 20
        alphas2 = np.linspace(0, 1, frames2 + 2)[1:]
        new_pos2, new_rot2 = [], []

        for a in alphas2:
            ipos = positions[-1].copy()
            irot = Apose_rot.copy()

            for side in raised_sides:
                arm_slice = arm_rotation_slices[side]
                arm_indices = range(arm_slice.start, arm_slice.stop)
                for joint_idx in arm_indices:
                    q0 = R.from_euler(
                        order, first_target_rot[joint_idx], degrees=True
                    ).as_quat()
                    q1 = R.from_euler(
                        order, Apose_rot[joint_idx], degrees=True
                    ).as_quat()
                    dot = np.dot(q0, q1)
                    if dot < 0:
                        q1 = -q1
                        dot = -dot
                    if dot > 0.9995:
                        q = q0 * (1-a) + q1 * a
                    else:
                        theta = np.arccos(dot)
                        sin_theta = np.sin(theta)
                        q = (
                            np.sin((1-a)*theta) / sin_theta * q0
                            + np.sin(a*theta) / sin_theta * q1
                        )
                    irot[joint_idx] = R.from_quat(q).as_euler(
                        order, degrees=True
                    )

            new_pos2.append(ipos)
            new_rot2.append(irot)

        positions = np.concatenate([positions, np.array(new_pos2)], axis=0)
        rotations = np.concatenate([rotations, np.array(new_rot2)], axis=0)

    #将qmai数据集转变为lafan格式
    names, offsets, parents, rotations, positions = qmai_to_lafan(names, offsets, parents, rotations, positions)

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)

def qmai_to_lafan(names, offsets, parents, rotations, positions):
    #将qmai的bvh调整成lafan格式，名称映射即删除多余骨骼（如手指）

    #创建lafan和seed的名称对照表,并替换
    SEED_TO_LAFAN_MAPPING = {
    'hips': 'Hips',
    'Chest': 'Spine',
    'Chest2': 'Spine1',
    'Chest3': 'Spine2',

    'LeftCollar': 'LeftShoulder',
    'LeftShoulder': 'LeftArm',
    'LeftElbow': 'LeftForeArm',
    'LeftWrist': 'LeftHand',
    'RightCollar': 'RightShoulder',
    'RightShoulder': 'RightArm',
    'RightElbow': 'RightForeArm',
    'RightWrist': 'RightHand',

    'LeftHip': 'LeftUpLeg',
    'LeftKnee': 'LeftLeg',
    'LeftAnkle': 'LeftFoot',
    'LeftToe': 'LeftToeBase',
    'RightHip': 'RightUpLeg',
    'RightKnee': 'RightLeg',
    'RightAnkle': 'RightFoot',
    'RightToe': 'RightToeBase',
    }
    
    names = [SEED_TO_LAFAN_MAPPING.get(item, item) for item in names]
    # print(f"映射后的名称表：{names}")

    lafan_default_names = ['Hips', 
                     'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                     'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
                     ]
    
    #通过原始names和lafan的默认names生成保留序列
    lafan_index = []
    for i, name in enumerate(names):
        if name in lafan_default_names:
            lafan_index.append(i)

    #去除offsets，rotations，positions的多余项
    lafan_offsets = offsets[lafan_index]
    lafan_rotations = rotations[:, lafan_index]
    lafan_positions = positions[:, lafan_index]

    #先将parents索引翻译为夫节点名称列表，去除对应项后再转回索引
    parents_names = []
    for i in parents:
        if i != -1:
            parents_names.append(names[i])
        else:
            parents_names.append("-1")
    # print(f"删减前的父节点名称表{parents_names}")
    lafan_parents_names = [parents_names[i] for i in lafan_index]
    # print(f"删减后的父节点名称表{lafan_parents_names}")
    lafan_parents = []
    for name in lafan_parents_names:
        if name != "-1":
            lafan_parents.append(lafan_default_names.index(name))
        else:
            lafan_parents.append(-1)
    
    return lafan_default_names, lafan_offsets, lafan_parents, lafan_rotations, lafan_positions
