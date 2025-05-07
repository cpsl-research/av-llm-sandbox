import math
from enum import Enum
from typing import List

from avstack.environment.objects import VehicleState
from avstack.geometry.transformations import transform_orientation


def get_all_meta_actions(
    agent_current: VehicleState,
    agent_future: VehicleState,
) -> List[str]:
    return [ACTION.evaluate(agent_current, agent_future) for ACTION in ACTIONS]


class Lateral(int, Enum):

    TURN_LEFT = -3
    CHANGE_LANE_LEFT = -2
    VEER_LEFT = -1
    STRAIGHT = 0
    VEER_RIGHT = 1
    CHANGE_LANE_RIGHT = 2
    TURN_RIGHT = 3

    @staticmethod
    def evaluate(
        agent_current,
        agent_future,
        thresh_veer: float = 5,
        thresh_turn: float = 20,
        positive_is_left: bool = True,
    ) -> int:
        """Determine if the vehicle goes left, straight or right

        Args:
            thresh_veer - threshold for defining a veer in degrees
            thresh_turn - threshold for defining a turn in degrees
            positive_is_left - if true, the positive delta yaw is a left turn
        """

        # just evaluate the attitude
        att_diff = agent_future.attitude.change_reference(
            agent_current.as_reference(), inplace=False
        )

        # test the resulting yaw angle - changing to positive is left convention
        d_yaw = 180 / math.pi * transform_orientation(att_diff.q, "quat", "euler")[2]
        if not positive_is_left:
            d_yaw *= -1
        if abs(d_yaw) >= thresh_turn:
            action = Lateral.TURN_LEFT if d_yaw > 0 else Lateral.TURN_RIGHT
        elif abs(d_yaw) >= thresh_veer:
            action = Lateral.VEER_LEFT if d_yaw > 0 else Lateral.VEER_RIGHT
        else:
            action = Lateral.STRAIGHT

        # TODO: we don't have change lanes yet

        return action


class Longitudinal(int, Enum):

    REVERSE = -3
    BRAKE_TO_STOP = -2
    DECEL = -1
    MAINTAIN = 0
    ACCEL = 1

    @staticmethod
    def evaluate(
        agent_current,
        agent_future,
        thresh_change: float = 0.25,
        thresh_stop: float = 0.50,
    ) -> int:
        """Determine if the vehicle decelerates, maintains, or accelerates

        Args:
            thresh_change - threshold to determine a change in velocity
            thresh_stop - threshold on velocity to call "stopped"
        """

        # just evaluate the velocity
        speed_diff = agent_future.velocity.norm() - agent_current.velocity.norm()

        # TODO: determine if we are reversing

        # test the resulting speed differential
        if speed_diff >= thresh_change:
            action = Longitudinal.ACCEL
        elif speed_diff <= -thresh_change:
            if agent_future.velocity.norm() <= thresh_stop:
                action = Longitudinal.BRAKE_TO_STOP
            else:
                action = Longitudinal.DECEL
        else:
            action = Longitudinal.MAINTAIN

        return action


ACTIONS = [
    Lateral,
    Longitudinal,
]
