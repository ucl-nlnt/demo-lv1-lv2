from fastapi import FastAPI
webserver = FastAPI()

@webserver.get("/")
def read_root():
    return {"Hello": "World"}


@webserver.get("/metadata")
def read_metadata():
    total_distance_traveled = 0
    total_degrees_rotated = 0
    return {"total_distance_traveled": total_distance_traveled, "total_degrees_rotated": total_degrees_rotated}