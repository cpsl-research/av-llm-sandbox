import json
import os
from argparse import ArgumentParser

import numpy as np
from avapi.nuscenes import nuScenesManager
from avstack.geometry import q_mult_vec, q_stan_to_cam
from avstack.geometry.transformations import project_to_image, transform_orientation
from tqdm import tqdm

from avlm.actions import ACTIONS, Lateral, Longitudinal


def main(args):
    if len(args.output_prefix) == 0:
        raise ValueError("Output prefix is empty, provide something like 'dataset'")

    # parse dataset name
    if args.dataset.lower() == "nuscenes":
        SM = nuScenesManager(
            data_dir=args.dataset_path,
            split=args.version,
        )
    elif args.dataset.lower() == "carla":
        raise NotImplementedError
    else:
        raise NotImplementedError(args.dataset.lower())

    # build metadata
    metadata = {
        "action_table": {
            action.__name__: {str(state): int(state) for state in action}
            for action in ACTIONS
        },
        "reverse_action_table": {
            action.__name__: {int(state): str(state) for state in action}
            for action in ACTIONS
        },
        "dataset": args.dataset,
        "waypoints_3d_reference": "camera",
    }

    # loop over splits
    for split in ["train", "val", "test"]:
        frames_completed = 0
        agents_completed = 0
        scenes_completed = 0
        scenes = SM.splits_scenes[split]
        print(f"\n\nProcessing split: {split}, {len(scenes)} scenes\n\n")

        # loop over available scenes
        ds_scenes = {}
        for i_scene, scene in enumerate(scenes):
            print(f"Processing scene {i_scene+1}/{len(scenes)}")
            ds_agents = {}
            try:
                SD = SM.get_scene_dataset_by_name(scene)
            except Exception:
                print(
                    f"Scene {scene} could not be loaded, potentially due to missing CAN data (for nuScenes)...skipping"
                )
                continue

            # loop over available agents
            agents = SD.get_agent_set(frame=0)
            for i_agent, agent in enumerate(agents):
                print(f"Processing agent {i_agent+1}/{len(agents)}")
                ds_agent = {}
                agent_state_init = None
                agent_state_last = None

                # loop over frames
                timestamps_all = SD.get_timestamps(
                    sensor=None, agent=agent, utime=False
                )
                frames_all = SD.get_frames(sensor=None, agent=agent)
                for frame in tqdm(frames_all):
                    sensor_primary = (
                        "main_camera"  # assume all cameras are nearly synched
                    )

                    #########################################################
                    # NOTE: I am not 100% sure that changing frames
                    # correctly accounts for the velocity and acceleration
                    # fields yet, so this is something to watch out for
                    #########################################################

                    # get agent information
                    timestamp = SD.get_timestamp(
                        frame=frame, sensor=sensor_primary, agent=agent
                    )
                    cam_calib = SD.get_calibration(
                        frame=frame, sensor=sensor_primary, agent=agent
                    )
                    # -- global frame is with world origin
                    agent_state_global = SD.get_agent(frame=frame, agent=agent)
                    agent_state_reference = agent_state_global.as_reference()
                    if agent_state_init is None:
                        agent_state_init = agent_state_global
                    # -- local frame is with t=0 origin
                    agent_state_local = agent_state_global.change_reference(
                        agent_state_init.as_reference(),
                        inplace=False,
                    )
                    # -- diff frame is differential from last (TODO: fix this)
                    if agent_state_last is None:
                        agent_state_diff = agent_state_global.change_reference(
                            agent_state_global.as_reference(),  # change with itself, so should be 0's
                            inplace=False,
                        )
                    else:
                        agent_state_diff = agent_state_global.change_reference(
                            agent_state_last.as_reference(),
                            inplace=False,
                        )
                        agent_state_last = agent_state_global

                    # get camera images
                    camera_image_paths = {
                        sensor: SD.get_sensor_data_filepath(
                            frame=frame, sensor=sensor, agent=agent
                        )
                        for sensor in SD.get_sensor_names_by_type(
                            sensor_type="camera", agent=agent
                        )
                    }

                    # get future waypoints
                    dt_waypoints = 0.5  # spacing between waypoints
                    int_waypoints = 3  # total time interval
                    waypoints_3d = {}
                    waypoints_pixel = {}
                    for dt_ahead in np.arange(
                        dt_waypoints, dt_waypoints + int_waypoints, dt_waypoints
                    ):
                        if timestamp + dt_ahead <= timestamps_all[-1]:
                            # get the frame corresponding to this time
                            frame_ahead = SD.get_frame_at_timestamp(
                                timestamp=timestamp + dt_ahead,
                                sensor=sensor_primary,
                                agent=None,
                                utime=False,
                                dt_tolerance=0.5,
                            )

                            # get future agent position for 3d waypoints in cam coordinates
                            agent_future = SD.get_agent(frame=frame_ahead, agent=agent)
                            box_future = agent_future.box.change_reference(
                                agent_state_reference, inplace=False
                            )
                            waypoint_3d = q_mult_vec(
                                q_stan_to_cam,
                                box_future.position.x,
                            )

                            # convert to pixel coordinates in camera
                            position_future_camera = (
                                agent_future.position.change_reference(
                                    cam_calib.reference, inplace=False
                                )
                            )
                            waypoint_pixel = project_to_image(
                                position_future_camera[:, None].T, cam_calib.P
                            )[0, :]
                        else:
                            waypoint_3d = None
                            waypoint_pixel = None

                        # store waypoints
                        waypoints_3d[f"dt_{dt_ahead:.2f}"] = (
                            list(waypoint_3d)
                            if waypoint_3d is not None
                            else waypoint_3d
                        )
                        waypoints_pixel[f"dt_{dt_ahead:.2f}"] = (
                            list(waypoint_pixel)
                            if waypoint_pixel is not None
                            else waypoint_pixel
                        )

                    # get future meta actions
                    dt_action = 1
                    int_action = 3
                    meta_actions = {}
                    has_future_in_scene = False
                    for dt_ahead in np.arange(
                        dt_action, dt_action + int_action, dt_action
                    ):
                        # only run if the future time is within the dataset
                        if timestamp + dt_ahead <= timestamps_all[-1]:
                            has_future_in_scene = True
                            # get the frame corresponding to this time
                            frame_ahead = SD.get_frame_at_timestamp(
                                timestamp=timestamp + dt_ahead,
                                sensor=sensor_primary,
                                agent=None,
                                utime=False,
                                dt_tolerance=0.5,
                            )

                            # get the meta action defined at this time
                            agent_current = agent_state_global
                            agent_future = SD.get_agent(frame=frame_ahead, agent=agent)
                            meta_action = {
                                "lateral": str(
                                    Lateral.evaluate(agent_current, agent_future)
                                ),
                                "longitudinal": str(
                                    Longitudinal.evaluate(agent_current, agent_future)
                                ),
                            }

                        else:
                            meta_action = None

                        # store results
                        meta_actions[f"dt_{dt_ahead:.2f}"] = meta_action

                    # store all data for this frame
                    ds_frame = {
                        "scene": scene,
                        "agent": agent,
                        "frame": frame,
                        "timestamp": timestamp,
                        "image_paths": camera_image_paths,
                        "meta_actions": meta_actions,
                        "has_future_in_scene": has_future_in_scene,
                        "waypoints_3d": waypoints_3d,
                        "waypoints_pixel": waypoints_pixel,
                        "agent_state": {
                            view: {
                                "position": list(state.position.x),
                                "velocity": list(state.velocity.x),
                                "speed": state.velocity.norm(),
                                # f"acceleration_{view}: list(state.acceleration.x),
                                # NOTE: accel may not be working
                                "attitude": [
                                    state.attitude.q.w,
                                    state.attitude.q.x,
                                    state.attitude.q.y,
                                    state.attitude.q.z,
                                ],
                                "yaw": transform_orientation(
                                    state.attitude.q, "quat", "euler"
                                )[2],
                            }
                            for view, state in zip(
                                ["global", "local", "diff"],
                                [
                                    agent_state_global,
                                    agent_state_local,
                                    agent_state_diff,
                                ],
                            )
                        },
                    }

                    # add this frame to the agent
                    ds_agent[f"frame_{frame}"] = ds_frame
                    frames_completed += 1

                # add for this agent
                ds_agents[f"agent_{agent}"] = ds_agent
                agents_completed += 1

            # add for this scene
            ds_scenes[f"scene_{i_scene}"] = ds_agents
            scenes_completed += 1

        # add for this scene
        ds_split = {
            "dataset": ds_scenes,
            "metadata": metadata,
        }

        # print out results
        sep = "-" * 50
        print(
            f"\nFinished generating {split} dataset! Results:"
            f"\n\n{scenes_completed:>6d} scenes completed\n{agents_completed:>6d}"
            f" agents completed\n{frames_completed:>6d} frames completed\n\n{sep}"
        )

        # save dataset
        file_out = os.path.join(f"{args.output_prefix}_{split}.json")
        if len(os.path.dirname(file_out)) > 0:
            os.makedirs(os.path.dirname(file_out), exist_ok=True)
        with open(file_out, "w") as f:
            json.dump(ds_split, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument(
        "--version", type=str, required=True, choices=["v1.0-mini", "v1.0-trainval"]
    )
    parser.add_argument("--output_prefix", default="dataset", type=str)
    parser.add_argument("--dataset", default="nuscenes", type=str)
    args = parser.parse_args()
    main(args)
