import os
import glob
import pickle
import matplotlib.pyplot as plt
import imageio


class Trace(object):
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.game_score = None
        self.length = 0
        self.states = []


class State(object):
    def __init__(self, name, obs, action_vector, feature_vector, img):
        self.observation = obs
        self.image = img
        self.observed_actions = action_vector
        self.name = name
        self.features = feature_vector

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        return


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def create_video(path):
    video_dir = os.path.join(path, "video")
    try:
        os.makedirs(video_dir)
    except:
        clean_dir(video_dir)
    fileList = [path + "/" + x for x in sorted(os.listdir(path))]
    fileList.remove(video_dir)
    writer = imageio.get_writer(video_dir + "/" + 'video.mp4', fps=10)

    for im in fileList:
        writer.append_data(imageio.imread(im))
    writer.close()


def find_features_layer(agent):
    agent_model_layers = agent.get_parameter_list()
    for layer in agent_model_layers:
        if layer.shape.ndims == 1:
            return layer.name


mapped_actions = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [2],
    11: [3],
    12: [4],
    13: [5],
    14: [3],
    15: [4],
    16: [3],
    17: [4],
}

"""Atari env action meaning"""
# ACTION_MEANING = {
#     0: "NOOP",
#     1: "FIRE",
#     2: "UP",
#     3: "RIGHT",
#     4: "LEFT",
#     5: "DOWN",
#     6: "UPRIGHT",
#     7: "UPLEFT",
#     8: "DOWNRIGHT",
#     9: "DOWNLEFT",
#     10: "UPFIRE",
#     11: "RIGHTFIRE",
#     12: "LEFTFIRE",
#     13: "DOWNFIRE",
#     14: "UPRIGHTFIRE",
#     15: "UPLEFTFIRE",
#     16: "DOWNRIGHTFIRE",
#     17: "DOWNLEFTFIRE",
# }
