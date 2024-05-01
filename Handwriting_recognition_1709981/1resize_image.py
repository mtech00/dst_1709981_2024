from PIL import Image

image = Image.open('./datasets/realdata_handwriting.png')
print(f"Original size : {image.size}")

sunset_resized = image.resize((512, 512))
sunset_resized.save('./output/realdata_handwriting_resized.png')
 
 
