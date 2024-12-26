import argparse

import gymnasium as gym
import numpy as np
import torch
import imageio

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="none",
        reward_mode="none",
        num_envs=3,
        render_mode='rgb_array',
        sensor_configs=sensor_configs,
    )

    points = []
    gifs = []
    obs, _ = env.reset(seed=args.seed)
    gifs.append(env.render().cpu().numpy())

    pcd = env.gen_scene_pcd(include_robot=False)
    points.append(pcd)

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        pcd = env.gen_scene_pcd(include_robot=False)
        print(pcd.shape)
        points.append(pcd)
        gifs.append(env.render().cpu().numpy())
        if torch.all(terminated) or torch.all(truncated):
            break
    env.close()
    np.save('points.npy', np.asarray(points, dtype=object))
    gifs = np.stack(gifs, axis=1) # num_env, t, h, w, 3
    for num_envs in range(gifs.shape[0]):
        imageio.mimsave(f'sample_{num_envs}.gif', gifs[num_envs])

if __name__ == "__main__":
    main(parse_args())
