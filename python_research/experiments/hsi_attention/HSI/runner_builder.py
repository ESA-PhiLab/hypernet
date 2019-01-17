
for i in range(3):
    for j in range(30):
        print("python ..\main.py --dataset pavia --validation 0.1 --test 0.1 --epochs 200 --patience 5 --modules {} --output pavia_{}_modules_run_{} --batch_size 200 --uses_attention True".format(i+2, i+2, j))

for i in range(3):
    for j in range(30):
        print(
            "python ..\main.py --dataset pavia --validation 0.1 --test 0.1 --epochs 200 --patience 5 --modules {} --output pavia_{}_modules_run_{}_no_attention --batch_size 200 --uses_attention False".format(
                i + 2, i + 2, j))


for i in range(3):
    for j in range(30):
        print("python ..\main.py --dataset salinas --validation 0.1 --test 0.1 --epochs 200 --patience 5 --modules {} --output salinas_{}_modules_run_{} --batch_size 200 --uses_attention True".format(i+2, i+2, j))

for i in range(3):
    for j in range(30):
        print(
            "python ..\main.py --dataset salinas --validation 0.1 --test 0.1 --epochs 200 --patience 5 --modules {} --output salinas_{}_modules_run_{}_no_attention --batch_size 200 --uses_attention False".format(
                i + 2, i + 2, j))