from PIL import Image

image = Image.open('./output/s1707272_0.jpg')
print(f"Original size : {image.size}") # 5464x3640

sunset_resized = image.resize((10, 10))
sunset_resized.save('./output/sunset_400.jpeg')
 
 
