from typing import Dict, List

from pydantic import Field, BaseModel


class DocumentIndex(BaseModel):
    task_id: str = Field(example="task-532")
    state_name: str = Field(example="California")
    input_dir: str = Field(example="test_data/json/files/")
    files: List[str] = Field(
        example=["www.fdle.state.fl.us_FSAC_UCR_2018_Counties_Volusia18.json"]
    )