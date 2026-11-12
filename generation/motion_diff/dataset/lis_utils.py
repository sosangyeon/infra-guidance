"""
lis_utils.py (dataset 로딩) — LIS 도메인용 데이터셋 클래스.

원본 `dataset/utils.py`의 TargetBuilder 기본값은 11/80(WOMD 91 step)으로 하드코딩돼 있고,
`womd.py`가 `TargetBuilder()`를 인자 없이 호출하므로 config의 step 값이 무시된다.
LIS는 40 step(historical 10 + future 30)이라 그대로 쓰면 TargetBuilder의
`assert hist+fut == T` 에서 크래시한다.

여기서는 원본을 수정하지 않고, LIS step에 맞춘 TargetBuilder를 주입한
`LISMotionDataset`을 별도로 제공한다. preprocess 로직(roadgraph_*/agent_*/attrs 읽기)은
WaymoMotionDataset 것을 그대로 상속한다 (h5 스키마가 동일하므로).

주의: 이 파일은 `dataset/process/lis_utils.py`(CSV/JSON 디코더)와 다른 모듈이다.
       (motion_diff.dataset.lis_utils vs motion_diff.dataset.process.lis_utils)

연결: data_module.py 가 cfg["dataset"]["name"] == "lis" 일 때 이 클래스를 사용.
"""

from typing import Optional

from .utils import TargetBuilder
from .womd import WaymoMotionDataset

# lis_config.yml 의 model.num_historical_steps / num_future_steps 와 반드시 일치해야 함
LIS_NUM_HISTORICAL_STEPS = 10
LIS_NUM_FUTURE_STEPS = 30


class LISMotionDataset(WaymoMotionDataset):
    """WaymoMotionDataset과 동일하나 LIS step(10/30)에 맞춘 TargetBuilder를 사용."""

    def __init__(
        self,
        root: str,
        split: str,
        data_len: Optional[int] = None,
        num_historical_steps: int = LIS_NUM_HISTORICAL_STEPS,
        num_future_steps: int = LIS_NUM_FUTURE_STEPS,
    ):
        super(LISMotionDataset, self).__init__(
            root=root,
            split=split,
            transform=TargetBuilder(
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
            ),
            data_len=data_len,
        )
