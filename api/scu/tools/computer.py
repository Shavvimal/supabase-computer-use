from typing import Literal, TypedDict, Optional, Union
from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult

Action = Literal[
    "type",
    "mouse_move",
    "left_click",
    # "left_click_drag",
    # "right_click",
    # "middle_click",
    # "double_click",
    # "screenshot",
    # "cursor_position",
]

class Resolution(TypedDict):
    width: int
    height: int

class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

class ComputerInstruction(TypedDict):
    action: Action
    text: Optional[str]
    coordinate: Optional[tuple[int, int]]
    wait_ms: Optional[int] # Wait time after action in milliseconds
    screenshot: Optional[str]  # Base64 encoded screenshot data

class ComputerTool(BaseAnthropicTool):
    """
    A tool that returns JSON instructions for browser-based computer interactions.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    def __init__(self, width: int = 1920, height: int = 1080, selected_screen: int = 0):
        super().__init__()
        self.width = width
        self.height = height
        self.display_num = selected_screen

    @property
    def options(self) -> ComputerToolOptions:
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        instruction: ComputerInstruction = {
            "action": action,
            "text": text,
            "coordinate": coordinate,
            "wait_ms": 50,
            "screenshot": None
        }

        print(instruction)

        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(f"{text} must be a string")

            # For typing actions, set a longer wait time
            instruction["wait_ms"] = 12 * len(text)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

        if action == "screenshot":
            instruction["wait_ms"] = 1000  # Give more time for screenshot capture

        return ToolResult(output=instruction)