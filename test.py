import torch
import torchvision
import torch.optim
import os
from modeling import model
import glob
import time
from option import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_VISIBLE_DEVICES']='0' # For GPU only
device = get_device()


def Tester(image_path, args, scale_factor=12):

	# Read image from path
	data_lowlight = image_from_path(image_path)

	# Scale image to have the resolution of multiple of 4
	data_lowlight = scale_image(data_lowlight, scale_factor, device) if scale_factor != 1 else data_lowlight

	# Build model
	EFE_net = model.enhance_net_nopool(scale_factor, conv_type='dsc').to(device) # TAKE care of this conv

	# Load weight directory
	EFE_net.load_state_dict(torch.load(args.weight_dir, map_location=device))

	# Run model inference
	start = time.time()
	enhanced_image,params_maps = EFE_net(data_lowlight)
	end_time = (time.time() - start)
	print(end_time)

	# Load result directory and save image
	image_path = image_path.replace('test_data',args.test_dir)
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	torchvision.utils.save_image(enhanced_image, result_path)

	return end_time

def test():
	with torch.no_grad():
		filePath = args.input_dir
		print(filePath)
		file_list = os.listdir(filePath)
		sum_time = 0

		for file_name in file_list:
			test_list = glob.glob(filePath + file_name + "/*")
			print(test_list)

		for image in test_list:
			print(image)
			sum_time = sum_time + Tester(image, args)

		print(sum_time)


if __name__ == '__main__':
	test()


		

