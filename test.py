import torch
import torchvision
import torch.optim
import os
from modeling import model
import glob
import time
from option import *
from utils import *
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES']='0' # For GPU only
device = get_device()

class Tester(): # TODO: replace image_path
	def __init__(self):
		self.scale_factor = 12
		self.net = model.enhance_net_nopool(self.scale_factor, conv_type='dsc').to(device)
		self.net.load_state_dict(torch.load(args.weight_dir, map_location=device))

	def inference(self, image_path):
		# Read image from path
		data_lowlight = image_from_path(image_path)

		# Scale image to have the resolution of multiple of 4
		data_lowlight = scale_image(data_lowlight, self.scale_factor, device) if self.scale_factor != 1 else data_lowlight

		# Run model inference
		start = time.time()
		enhanced_image, params_maps = self.net(data_lowlight)
		end_time = (time.time() - start)
		print(end_time)

		# Load result directory and save image
		image_path = image_path.replace('test_data', args.test_dir)
		result_path = image_path
		if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
			os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
		torchvision.utils.save_image(enhanced_image, result_path)

		return end_time

	def test(self):
		self.net.eval()
		filePath = args.input_dir
		print(filePath)
		file_list = os.listdir(filePath)
		sum_time = 0

		for file_name in file_list:
			test_list = glob.glob(filePath + file_name + "/*")
			print(test_list)

		for image in test_list:
			print(image)
			sum_time = sum_time + self.inference(image)

		print(sum_time)
		print("test finished!")





if __name__ == '__main__':
	t = Tester()
	t.test()



		

