import os

acc_list = []
results_total_file = open("results_total_file.txt",'w+')

for f_name in os.listdir("results"):
    f = f_name.split('.')
    if f[-1] == 'txt':
        f = f[0]+f[1]
        f = f.split('_')
        if f[0] == 'multilayer':
            results_total_file.write("\subsection{bias: " + f[1] +", activation: "+ f[2] + ", lr: 0."+f[3] +", epochs: "+ f[4] + "}\n\n")
            # print(f[1],f[2],"0."+f[3],f[4])
            with open("results/"+f_name,'r') as f_:
                file_contents = f_.read()
                for item in file_contents.split('\n'):
                    results_total_file.write(item+'\n\n')
                # print(file_contents)
                acc = file_contents.split('\n')[4].split(':')[1]
                acc_list.append(float(acc))
                # print(float(acc))

acc_list.sort()
print("highest accuracy:",acc_list[-1])