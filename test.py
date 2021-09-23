import torch
import torchvision
import torch.optim
import os
import argparse
from modeling import model
import numpy as np
from PIL import Image
import glob
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lowlight(image_path, args):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)


	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=((data_lowlight.shape[0])//scale_factor)*scale_factor
	w=((data_lowlight.shape[1])//scale_factor)*scale_factor
	print("cropped height is ", h)
	print("cropped width is", w)
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.to(device).unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor, conv_type='dsc').to(device) # TAKE care of this conv

	############ load weight directory  ############
	DCE_net.load_state_dict(torch.load(args.weight_dir, map_location=device))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	########## load result directory ###########
	image_path = image_path.replace('test_data',args.test_dir)

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--weight_dir', type=str, help='directory for pretrained weight')
	parser.add_argument('--test_dir', type=str, help='directory for testing output')
	args = parser.parse_args()

	with torch.no_grad():

		filePath = 'data/test_data/'
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
		for image in test_list:

				print(image)
				sum_time = sum_time + lowlight(image, args)

		print(sum_time)
		

