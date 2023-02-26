import os
class Config(object):
    DEBUG =False
    TESTING = False
    SECRET_KEY ="xVrzndRlxr"
    DB_NAME = "production.db"
    DB_USERNAME = "root"
    DB_PASSWORD = "root"

    #files upload dir path 
    UPLOAD = os.path.join(os.getcwd(),"app","static","uploads")
    MODELS = os.path.join(os.getcwd(),'app','static','models' )
    STATIC = os.path.join(os.getcwd(),'app','static' )
    SESSION_COOKIE_SECURE = True

    ENV = "production"
     

class ProductionConfig(Config):
    pass

class DeveolpmentConfig(Config):
    DEBUG =True

    DB_NAME = "dev.db"
    DB_USERNAME = "root"
    DB_PASSWORD = "root"

    SESSION_COOKIE_SECURE = False

    ENV="devolpment"

class TestingConfig(Config):
    TESTING =True

    DB_NAME = "test.db"
    DB_USERNAME = "root"
    DB_PASSWORD = "root"

    SESSION_COOKIE_SECURE = False

    ENV = "testin"
    

