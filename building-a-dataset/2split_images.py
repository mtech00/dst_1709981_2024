from split_image import split_image
split_image("./datasets_tp/tabela6_approved.png",10, 10, False, False,output_dir="./output/")
#split_image(image_path, rows, cols, should_square, should_cleanup, [output_dir])
# e.g. split_image("bridge.jpg", 2, 2, True, False)