# Video -> Image
python demo/make_video.py --choice V2I

# Image Inference (NOTE: specifiy these paths on your own)
python test.py --input_dir $image_lowlight_folder --test_dir $image_folder

# Image -> Video
python demo/make_video.py --choice I2V
