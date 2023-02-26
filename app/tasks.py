from app import app
# import time
import os 
import shutil
import requests
# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline, DDIMScheduler




def acc_model(model_name,output_dir,vae_name,resolution,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path,weights_dir_path,model_path,push_id):
    
    command_str = f"""accelerate launch ./app/train_dreambooth.py \
  --pretrained_model_name_or_path="{model_name}" \
  --pretrained_vae_name_or_path="{vae_name}" \
  --output_dir="{output_dir}" \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution={resolution} \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate={learning_rate} \
  --lr_scheduler="{lr_scheduler}" \
  --lr_warmup_steps={lr_warmup_steps}\
  --num_class_images={num_class} \
  --sample_batch_size=4 \
  --max_train_steps={train_step} \
  --save_interval=10000 \
  --save_sample_prompt="{save_sample_prompt}" \
  --instance_prompt="{instance_prompt}" \
  --class_prompt="{class_prompt}" \
  --instance_data_dir="{images_path}" \
  --class_data_dir="{photo_path}" 2>&1 | tee ./app/command.txt"""
    # print(command_str)
    with open('./app/command.txt','a') as f :
        f.write('**********************************************************************************\n')
    # time.sleep(10)
    result =  os.system(command_str)

    try :
        shutil.rmtree(images_path, ignore_errors=True)
    except : 
        pass
    try :
        shutil.rmtree(photo_path, ignore_errors=True)
    except :
        pass
    #@markdown Run conversion.
    ckpt_path = os.path.join(weights_dir_path , model_path+".ckpt")

    half_arg = ""
    #@markdown  Whether to convert to fp16, takes half the space (2GB).
    fp16 = True #@param {type: "boolean"}
    if fp16:
        half_arg = "--half"
    os.system(f'python ./app/convert_diffusers_to_original_stable_diffusion.py --model_path="{weights_dir_path}"  --checkpoint_path="{ckpt_path}" {half_arg}')
    shutil.copy(ckpt_path,app.config['MODELS'])
    shutil.rmtree(output_dir, ignore_errors=True)
    re_api = requests.post('https://api.iqstars.me/In2Niaga/Notification.aspx',json={"DeviceID":push_id})
    if re_api.json() and re_api.json()['message'] :
        if re_api.json()['message'] == "Success":
            print("****************Success*********************")
        else :
            print("*****************Faild**********************")


    return result




def queue_turn(model_name,output_dir,vae_name,resolution,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path,weights_dir_path,model_path, push_id,usr):
    mo =  acc_model(model_name,output_dir,vae_name,resolution,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path,weights_dir_path,model_path,push_id)
    # if "Error" not in mo :
    #     print("inf")
    
# else:
#     print("Error")
    
    # os.mkdir(secrets.token_urlsafe(6))
    
