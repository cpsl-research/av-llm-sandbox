import json
from argparse import ArgumentParser
from tqdm import tqdm

from avapi.nuscenes import nuScenesManager
from avstack.geometry.transformations import transform_orientation
from meta_actions import get_all_meta_actions


def main(args):
    dataset = {}

    if args.dataset.lower() == "nuscenes":
        SM = nuScenesManager(data_dir=args.dataset_path)
    elif args.dataset.lower() == "carla":
        raise NotImplementedError
    else:
        raise NotImplementedError(args.dataset.lower())

    # loop over splits
    for split in ["train", "val", "test"]:
        ds_split = {}
        scenes = SM.splits_scenes[split]
        print(f"\n\nProcessing split: {split}, {len(scenes)} scenes")

        # loop over available scenes
        for i_scene, scene in enumerate(scenes):
            print(f"Processing scene {i_scene+1}/{len(scenes)}")
            ds_scene = {}
            SD = SM.get_scene_dataset_by_name(scene)

            # loop over available agents
            agents = SD.get_agent_set(frame=0)
            for i_agent, agent in enumerate(agents):
                print(f"Processing agent {i_agent+1}/{len(agents)}")
                ds_agent = []
                agent_state_init = None
                agent_state_last = None

                # loop over frames
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
                    # -- global frame is with world origin
                    agent_state_global = SD.get_agent(frame=frame, agent=agent)
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

                    #########################################################
                    # NOTE: I am currently running this with some fixed
                    # number of frames ahead. This can be changed. It will
                    # not scale properly between datasets with different
                    # data rates. If there is not data for the desired
                    # look-ahead frame number, it says no meta action.
                    #########################################################

                    # get meta actions over some time horizon
                    frames_ahead = [1, 5, 10]
                    # time_ahead = [1]
                    meta_actions = {
                        ahead: get_all_meta_actions(
                            agent_current=agent_state_global,
                            agent_future=SD.get_agent(frame=frame + ahead, agent=agent),
                        )
                        if frame + ahead in frames_all
                        else None
                        for ahead in frames_ahead
                    }

                    # store all data for this frame
                    ds_agent.append(
                        {
                            "frame": frame,
                            "timestamp": timestamp,
                            "image_paths": camera_image_paths,
                            "meta_actions": meta_actions,
                            "agent_state": {
                                view: {
                                    f"position_{view}": list(state.position.x),
                                    f"velocity_{view}": list(state.velocity.x),
                                    f"speed_{view}": state.velocity.norm(),
                                    # f"acceleration_{view}: list(state.acceleration.x),
                                    # NOTE: accel may not be working
                                    f"attitude_{view}": [
                                        state.attitude.q.w,
                                        state.attitude.q.x,
                                        state.attitude.q.y,
                                        state.attitude.q.z
                                    ],
                                    f"yaw_{view}": transform_orientation(
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
                    )

                # add for this agent
                ds_scene[f"agent_{agent}"] = ds_agent

            # add for this scene
            ds_split[scene] = ds_scene

        # add for this split
        dataset[split] = ds_split

    # save dataset
    with open(args.file_out, "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("file_out", type=str)
    parser.add_argument("--dataset", default="nuscenes", type=str)
    args = parser.parse_args()
    main(args)
