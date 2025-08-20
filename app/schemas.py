from pydantic import BaseModel
class MoveCommand(BaseModel):
    command: str