import os





def get_file_names(data_dir):
    file_names = []
    nfold = 10


    for n in range(nfold):
        file_name = 'FDDB_OUTPUT/FDDB-det-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        file_names.append(file_name)
    #print(file_names)
    return file_names

def merge_files(file_names,output_file_name):
    output_file_name = output_file_name
    output_file = open(output_f_name,'w')
    for file_name in file_names:
        fid = open(file_name,mode = 'r')
        for line in fid:
            output_file.writelines(line)

        output_file.write('\n')
        print(file_name,' done')
    output_file.close()




if __name__ == '__main__':
    merge_file_dir = '../../DATA/'
    output_dir = '../../DATA/FDDB_OUTPUT'
    output_f_name = os.path.join(output_dir, 'FDDB-det-fold-all.txt')
    #output_f_name = os.path.join(output_dir, 'Fold-all.txt')
    file_names = get_file_names(merge_file_dir)
    merge_files(file_names,output_f_name)