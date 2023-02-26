from flask import Flask
import os
# from app import tasks,views,config
# print("********************************************************before init********************************************************")

app = Flask(__name__)
app.config["ENV"] =="production"
# print("********************************************************init********************************************************")
if app.config["ENV"] == "production"  :
    app.config.from_object("config.ProductionConfig")

elif app.config["ENV"] == "testin"  :
    app.config.from_object("config.TestingConfig")
else  :
    app.config.from_object("config.DeveolpmentConfig")

    # print("***********************************linux*****************")
import redis
from rq import Queue
# from rq.registry import StartedJobRegistry

# url="redis://redis:6379/0"
url= "redis://localhost:6379/0"
r = redis.from_url(url)
# r =redis.Redis(host='0.0.0.0', port=6379, decode_responses=True)
q=Queue(connection=r,default_timeout=21600)

# registry = StartedJobRegistry('default', connection=r)

from app import tasks,views,face_crop
