"""
This is actually shite. Will need to do better Auth than a string LOL
"""

import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException

_ = load_dotenv()

async def verify_token(x_token: str = Header()):
    pass
