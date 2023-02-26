# import os 
# print(os.listdir(os.path.join(os.getcwd(),"app")))
from app import app
# from gevent.pywsgi import WSGIServer

if __name__ == '__main__' :
    # print(os.listdir(os.path.join(os.getcwd(),"app")))
    
    app.config["ENV"] == "production" 
    # port = int(os.environ.get('PORT', 5050))
    # print("********************************************************before run********************************************************")
   
    # os.mkdir(os.path.join(app.config['CLIENT_FOLDER'],'admin'))
    app.run(host="0.0.0.0",port=5050)
    # http_server = WSGIServer(('', port), app)
    # http_server.serve_forever()

    # print("********************************************************run********************************************************")


    
    # print(app.config)
