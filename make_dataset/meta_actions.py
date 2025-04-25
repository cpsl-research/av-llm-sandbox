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

    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    @staticmethod
    def evaluate(
        agent_current,
        agent_future,
        thresh_turn: float = 30,
        positive_is_left: bool = True,
    ) -> int:
        """Determine if the vehicle goes left, straight or right

        Args:
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
        if d_yaw >= thresh_turn:
            action = Lateral.LEFT
        elif d_yaw <= -thresh_turn:
            action = Lateral.RIGHT
        else:
            action = Lateral.STRAIGHT

        return action


class Longitudinal(int, Enum):

    DECEL = -1
    MAINTAIN = 0
    ACCEL = 1

    @staticmethod
    def evaluate(
        agent_current,
        agent_future,
        thresh_change: float = 0.25,
    ) -> int:
        """Determine if the vehicle decelerates, maintains, or accelerates

        Args:
            thresh_change - threshold to determine a change in velocity
        """

        # just evaluate the velocity
        speed_diff = agent_future.velocity.norm() - agent_current.velocity.norm()

        # test the resulting speed differential
        if speed_diff >= thresh_change:
            action = Longitudinal.ACCEL
        elif speed_diff <= -thresh_change:
            action = Longitudinal.DECEL
        else:
            action = Longitudinal.MAINTAIN

        return action


ACTIONS = [
    Lateral,
    Longitudinal,
]
