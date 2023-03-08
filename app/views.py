from app import app 
from app.tasks import queue_turn
from app.face_crop import image_croping
from flask import  request,jsonify,make_response ,render_template ,redirect
from werkzeug.utils import secure_filename
import time
import datetime
import itertools
import os 
import secrets
import shutil
from app import r ,q

from rq.command import send_stop_job_command



def report_success(job, connection, result, *args, **kwargs):
    print("--------------------ssssssssssss------------------------")
    # print(job)
    print(connection)
    print(result)
    print("---------------------------------------------")

def report_failure(job, connection, type, value, traceback):
    print("--------------------ffffffffffffff----------------------")
    print(job)
    print(connection)
    print(type)
    print("---------------------------------------------")

if not os.path.isdir(app.config['MODELS']):
    os.mkdir(app.config['MODELS'])
if not os.path.isdir(app.config['UPLOAD']):
    os.mkdir(app.config['UPLOAD'])
HOME_PATH = os.path.join(os.getcwd(),'app')

JOBSIDS= []
@app.route("/")
def home():
    return "<h1>stable difiusion</h1>"

@app.route('/runwayml',methods=['POST'])
def yamlmodel():
    #args control ?
    if request.method == "POST" :
        args = request.json 
        usr=None
        if 'usr' in args :
            usr = args['usr']
        else:
            return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
        push_id = None
        if 'push_id' in args :
            push_id=args['push_id']
        else:
            return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
        
        images=[]
        if 'images' in args :
            if len(args['images']) > 1 :
                images= args['images']
            else:
                return make_response(jsonify({'status':"failed",'discription':'invalid images list'}),400)
        else:
            return make_response(jsonify({'status':"failed",'discription':'no images sent'}),400)
        
        # ["face","half","close","full"]
        cut_type = None#"face"
        if 'cut_type' in args :
            if args['cut_type'] in ["face","half","close","full"] :
                cut_type = args['cut_type']

        resolution = 512
        if 'resolution' in  args :
            try :
                resolution = int(args['resolution'])
            except :
                print("enter valid resolution")

        instance_prompt = "secourses"
        if 'instance_prompt' in  args :
            if len(args['instance_prompt'] ) > 2:
                instance_prompt = args['instance_prompt']
        
        class_prompt = "portrait photo of a person"
        if 'class_prompt' in  args :
            if len(args['class_prompt'] ) > 2:
                class_prompt = args['class_prompt']

        learning_rate = 1e-6
        if 'learning_rate' in  args :
            try :
                learning_rate = float (args['learning_rate'])
            except :
                print("enter valid learningrate")

        lr_scheduler = "constant"
        if 'lr_scheduler' in  args :
            if len(args['lr_scheduler'] ) > 2:
                lr_scheduler = args['lr_scheduler']
        
        lr_warmup_steps = 208
        if 'lr_warmup_steps' in  args :
            try :
                lr_warmup_steps = int(args['lr_warmup_steps'])
            except :
                print("enter valid lr_warmup_steps")

        num_class_images = 312
        if 'num_class_images' in  args :
            try :
                num_class_images = int(args['num_class_images'])
            except :
                print("enter valid num_class_images")
        
        max_train_steps = 2080
        if 'max_train_steps' in  args :
            try :
                max_train_steps = int(args['max_train_steps'])
            except :
                print("enter valid max_train_steps")

        save_sample_prompt = "secourses"
        if 'save_sample_prompt' in  args :
            if len(args['save_sample_prompt'] ) > 2:
                save_sample_prompt = args['save_sample_prompt']
        
        model_name = "runwayml/stable-diffusion-v1-5"
        if 'model_name' in  args :
            if len(args['model_name'] ) > 2:
                model_name = args['model_name']
    
        #######################################end param##########################################
        folder_name = secrets.token_urlsafe(6)

        # os.mkdir(os.path.join(app.config["UPLOAD"],folder_name))
        os.mkdir(os.path.join(app.config["UPLOAD"],folder_name +'input'))
        print("******************************")
        print(cut_type)
        images_number =image_croping(images,
                    os.path.join(app.config["UPLOAD"],folder_name+'input'),cut_on=cut_type)
        shutil.rmtree(os.path.join(app.config["UPLOAD"],folder_name), ignore_errors=True)

        model_path = 'custom'+'-_-'+str(usr)+'-_-'+str(datetime.datetime.now().strftime("%Y-%M-%d_%H-%M-%S"))+'-_-'+secrets.token_urlsafe(6)+'-_-'+secure_filename(model_name)+'-_-'+str(max_train_steps)
        output_dir =  os.path.join(app.config['MODELS'],model_path)
        weights_dir_path = os.path.join(output_dir,str(max_train_steps))
        
        arguments = {"images_path":os.path.join(app.config["UPLOAD"],folder_name+'input'),
                    "photo_path":os.path.join(app.config["STATIC"],"model_yaml","photos"),
                    "weights_dir_path":weights_dir_path,
                    "model_name":model_name,
                    "vae_name":"stabilityai/sd-vae-ft-mse",
                    "output_dir":output_dir,
                    "resolution":resolution,
                    "lr_scheduler":lr_scheduler,
                    "learning_rate":learning_rate,
                    "lr_warmup_steps":lr_warmup_steps,
                    "num_class":num_class_images,
                    "train_step":max_train_steps,
                    "save_sample_prompt":save_sample_prompt,############################
                    "instance_prompt":instance_prompt,
                    "class_prompt":class_prompt,
                    "model_path":model_path,
                    "push_id":push_id,
                    "usr":args['usr']}
        
        if images_number > 10 :
            job = q.enqueue(queue_turn,**arguments,on_success=report_success, on_failure=report_failure,result_ttl=21600)

            JOBSIDS.append(job.id)
            if len(JOBSIDS)>500 :
                JOBSIDS [len(JOBSIDS)-500:]

            return make_response(jsonify({'status':"Success","discription":f"job enterd to the queue with total images {images_number}",
            "weights_dir_path":model_path}))
        else :

            return make_response(jsonify({'status':"failed",'discription':" images less than 10"}),400)
    
    return make_response(jsonify({'status':"failed",'discription':'Method Not Allowed'}),405)

@app.route("/jobs-api",methods=['GET'])
def jobs_api():
    if request.method == "GET" :
        args = request.args 
        usr=None
        if 'usr' in args :
            usr=args['usr']
        else:
            return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
        
        batch=5
        if 'batch' in args :
            try :
                batch=int (args['batch'])
            except:
                return make_response(jsonify({'status':"failed",'discription':'enter valid batch'}),400)    
        # else:
        #     return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
        ########################################################################
        jobs = {}   
        counter = 0
        for jj in JOBSIDS:
            job = q.fetch_job(jj)
            if job is not None:
                if  job.kwargs['usr'] == usr :
                    counter = counter +1
                    jl = {"status":job.get_status(),
                          "model_id":job.kwargs["model_path"],
                          "enqueued_at":job.enqueued_at.strftime('%m/%d/%Y %H:%M:%S')}
                    if job.started_at is not None:
                        jl.update({"started_at":job.started_at.strftime('%m/%d/%Y %H:%M:%S')})
                    else :
                        pass
                    if  job.ended_at is not None :
                        jl.update({"ended_at":job.ended_at.strftime('%m/%d/%Y %H:%M:%S')})
                    else :
                        pass
                    jobs.update({"job_"+str(counter) :jl})
                else :
                    continue
            else:
                continue
        # if len(jobs) ==0 :
        #     return make_response(jsonify({'message':"success",'jobs':jobs[:batch]}),200)
        # else :
        return make_response(jsonify({'message':"success",'jobs':dict(list(jobs.items())[-1*batch:])}),200)
    else:
        return make_response(jsonify({'message':"failed",'discription':'Method Not Allowed'}),405)
    

@app.route('/cancel-job',methods=['POST'])
def cancel_job ():
    if request.method == "POST" :
        print("**************************************")
        print(request.json)
        print("**************************************")
        if request.json:
            args = request.json 
            usr=None
            if 'usr' in args :
                usr=args['usr']
            else:
                return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
            
            model_name=None
            if 'model_name' in args :
                try :
                    model_name=args['model_name']
                except:
                    return make_response(jsonify({'status':"failed",'discription':'enter valid batch'}),400)    
            else:
                return make_response(jsonify({'status':"failed",'discription':'no user'}),400)
            ########################################################################
            for jj in JOBSIDS:
                    job = q.fetch_job(jj)
                    if job is not None:
                        if  job.kwargs['usr'] == usr and job.kwargs["model_path"] == model_name:
                            job_id = job.id
                            if job_id in q.job_ids  :
                                
                                try :
                                    job.cancel()
                                    
                                except:
                                    return  make_response(jsonify({'message':"failed",'discription':'faild to cancel the job'}),400)
                            else :
                                try:
                                    send_stop_job_command(r, job_id)
                                except:
                                    return make_response(jsonify({'message':"failed",'discription':'job not found'}),400)

                            return make_response(jsonify({'message':"Success"}),200)
                        
            return make_response(jsonify({'message':"failed",'discription':'job not found'}),400)
        else :
            return make_response(jsonify({'message':"failed",'discription':'no json passed'}),400)
    else:
        return make_response(jsonify({'message':"failed",'discription':'Method Not Allowed'}),405)

@app.route("/jobs",methods=['GET'])
def wor():
    jobs = []   
    for jj in JOBSIDS:#q.job_ids :
        job = q.fetch_job(jj)
        if job is not None:
            jl = [jj, job.get_status(),job.enqueued_at.strftime('%m/%d/%Y %H:%M:%S'),]
            if job.started_at is not None:
                jl.append(job.started_at.strftime('%m/%d/%Y %H:%M:%S'))
            else :
                jl.append(None)
            if  job.ended_at is not None :
                jl.append(job.ended_at.strftime('%m/%d/%Y %H:%M:%S'))
            else :
                jl.append(None)
            if  job.kwargs['usr'] is not None :
                jl.append(job.kwargs['usr'])
            else :
                jl.append(None)
            if  job.kwargs['method'] is not None :
                jl.append(job.kwargs['method'])
            else :
                jl.append(None)

            jl.append(job.result)
            jobs.append(jl)
        
    return render_template("public/log_table.html",log_table=jobs)

@app.route('/cancel',methods=['GET'])
def cancel ():
    if 'id' in request.args:
        
        if request.args['id'] in q.job_ids  :
            
            try :
            
                job = q.fetch_job(request.args['id'])
                job.cancel()
                
            except:
                print("faild to cancel the job")
        else :
            try:
                send_stop_job_command(r, request.args['id'])
                time.sleep(2)
                return redirect("/jobs")
            except:
                print("job not found")
    else :
        print("no id sent")

    return redirect("/jobs")

# @app.route('/listmodels',methods=["GET"])
# def listmodels():
#     # args control ?
#     args = request.args 
    
#     if 'usr' in args :
#         if args['usr'] in os.listdir(app.config['USERS']):

#             user_folder = os.path.join(app.config['USERS'],args['usr'],'models' )
#             if os.path.isdir(user_folder):
#                 models = os.listdir(user_folder)
#                 return make_response(jsonify({'message':True,'models':models}),200)
#             else:
#                 return make_response(jsonify({'message':False,'discription':'user has not build any model yet'}),400)    
                
#         else:
#             return make_response(jsonify({'message':False,'discription':'user dose not exist'}),400)
#     else:
#         return make_response(jsonify({'message':False,'discription':'no user'}),400)

