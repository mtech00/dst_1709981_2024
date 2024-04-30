import os

path = "./output_text/"
dir_list = os.listdir(path)
count=0
dataset=""
count==-1
content=""

print(dir_list[0])
x=dir_list[0]
u="./output_text/"+x
content_final=""
≈
def build(contents,u):
    global dataset;
    conteudo_aux0 = contents.replace("\n", "")
    conteudo_aux1 = conteudo_aux0.replace(",", ";")
    conteudo_aux11 = conteudo_aux1.replace(" ≈ ", ",")
    conteudo_aux2_1 = conteudo_aux11.replace(" ", ",")
    conteudo_aux2_2 = conteudo_aux2_1.replace(",,", ",")
    conteudo_aux2_3 = conteudo_aux2_2.replace("[,", "[")
    conteudo_aux2 = conteudo_aux2_3.replace(",]", "]")
    w_array = conteudo_aux2.split(";")
    print("******************")

    lw_array=len(w_array)
    print("conteudo_aux2= ",u,conteudo_aux2)
    for w in w_array:
        w1=w.replace("[","")
        w2 = w1.replace("]", "")
        w3=w2.split(",")
        print(w3[0], ",", w3[1], ",", w3[2],"\n")
        print(w3[0],",",w3[1],",",w3[2],",",str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3)))
        dataset=dataset+";"+str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3))

for u in dir_list:
    v="./output_text/" + u
    with open(v) as f:
        content = f.readlines()
        for contents in content:
            build(contents,u)
    dataset=dataset+"\n"


h = open("./dataset_texto/dataset_final"+x, "a")
h.write(dataset)
h.close()
print("Files and directories in '", path, "' :")

print(dir_list)
