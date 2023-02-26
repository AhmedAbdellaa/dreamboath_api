from app import app
import os
# from gevent.pywsgi import WSGIServer

if __name__ == '__main__' :

    # print("********************************************************before run********************************************************")

    # print("****************wsgi*******************")
    # os.mkdir(os.path.join(app.config['CLIENT_FOLDER'],'admin'))
    app.run()