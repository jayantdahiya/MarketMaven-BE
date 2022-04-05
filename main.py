from imp import reload
import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", reload=True)