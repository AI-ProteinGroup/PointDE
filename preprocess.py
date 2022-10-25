import os
from re import L
import numpy as np
import operator
from functools import reduce
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import trange


N = 1000
processor = 24

input_dir = '/home/cxhrzh/chenzihao/dockground61'
output_dir = '/home/cxhrzh/dockground61_1000'

def get_atom_list_from_pdb(chain,chain_type):
    """
    Input:
        chain: input receptor/ligrand atoms (DataFram.Series)
        chain_type: 'R' or 'L'
    Return:
        atom_list: list of all atom features (List)
    """
    choice = {'R':'receptor','L':'ligand'}
    ATOM_TYPE = ['C','N','O']
    atom_list = []
    for index, atom in chain.iterrows():
        #print(index)
        _atom= atom.values.tolist()
        #x, y, z = [ float(x) for x in [atom.x,atom.y,atom.z]]
        atom_name = str(_atom[2]).strip()
        resName = str(_atom[3]).strip()
        resID = str(_atom[5]).strip()
        x = float(str(_atom[6]).strip())
        y = float(str(_atom[7]).strip())
        z = float(str(_atom[8]).strip())
        #print(x,y,z)
        if atom_name not in ATOM_TYPE:
            atom_name = 'Others'
        atom_tuple = [x,y,z,atom_name,resName,resID,chain_type,index]
        atom_list.append(atom_tuple)
    
    print('in {} has {} atoms'.format(choice[chain_type],len(atom_list)))
    return atom_list

def get_interface_from_atom_N(atom_r_list,atom_l_list, N=500):
    """
    Input:
        atom_1_list: list of all receptor atom features (List)
        atom_2_list: list of all ligrand atom features (List)
        N: cut off atoms for each chains
    Return:
        [r_list,l_list]: list of cutoff chain's atom features (List)
    """
    r_atom_list_ID =[]
    l_atom_list_ID =[]
    
    distances = []
    for r_ID , r_atom in enumerate(atom_r_list):
        for l_ID , l_atom in enumerate(atom_l_list):
            point_a= np.array(r_atom[0:3])
            point_b = np.array(l_atom[0:3])
            distance = np.sqrt(np.sum(np.square(point_a-point_b)))
            distances.append([r_ID,l_ID,distance])
    
    distances = np.array(distances)
    sort_distances = distances[np.argsort(distances[:,2])].tolist()

    num = 0
    for d in sort_distances:
        r_ID, l_ID, distance = d
        r_ID = int(r_ID)
        l_ID = int(l_ID)
        if (r_ID not in r_atom_list_ID) and (l_ID not in l_atom_list_ID):
            r_atom_list_ID.append(r_ID)
            l_atom_list_ID.append(l_ID)
            num = num+1
        if num == N:
            break

    r_list = []
    l_list = []

    for index in r_atom_list_ID:
        r_atom = atom_r_list[index]
        r_list.append(r_atom)
    for index in l_atom_list_ID:
        l_atom = atom_l_list[index]
        l_list.append(l_atom)

    print('after cutoff receptor has {} atoms'.format(len(r_list)))
    print('after cutoff ligand has {} atoms'.format(len(l_list)))
    return r_list,l_list
    

def encode_res(new_data, enc, pca=None):
    "one-hot encode residual type"
    new_data = np.array(new_data).reshape(len(new_data), -1)
    new_data = enc.transform(new_data).toarray()
    if pca:
        new_data = pca.transform(new_data)
    return new_data

def encode_atom_list(r_list,l_list,is_atom_list=False):
    """
    Input:
        r_list: list of cutoff receptor atom features (List)
        l_list: list of cutoff ligrand atom features (List)
        is_atom_list: for residual List
    Return:
        [r_list,l_list]: list of cutoff chain's atom features (List)
    """
    res_list = []

    if is_atom_list:
        res_list = r_list + l_list
    else:
        res_list = reduce(operator.add, r_list) + reduce(operator.add, l_list)
    
    atom_names_origin = ['C','N','O','Others']
    atom_names = np.array(atom_names_origin).reshape(len(atom_names_origin), -1)
    
    res_names_origin = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    res_names = np.array(res_names_origin).reshape(len(res_names_origin), -1)

    #One Hot
    res_enc = OneHotEncoder()
    res_enc.fit(res_names)
    
    atom_enc = OneHotEncoder()
    atom_enc.fit(atom_names)
    
    #Encode
    new_list =[]
    for row in res_list:
        xyz = np.array(row[0:3])
        resName = encode_res([row[4]],res_enc)[0]
        atomName =encode_res([row[3]],atom_enc)[0]
        chain = row[-1]
        if chain=="R":
            chain = np.array([1,0])
        else:
            chain = np.array([0,1])
        new_row = np.concatenate([xyz,atomName,resName,chain])
        new_list.append(new_row)
    return new_list

def process_pdb_file_by_atom_N(input_pdb_file,sv_pdb_file,chian_id_1='A',chian_id_2='B',fix_atom_num=1024):
    """
    Input:
        input_pdb_file: input protein file (.pdb type)
        sv_pdb_file: output protein file (.pdb type)
        chian_id_1: protein receptor chain name list, default=['A']
        chian_id_2: protein ligrand chain name list, defualt=['B']
        fix_atom_num: fix atom number
    Return:
        e_list: an encoder atom list, inclue [x,y,z,one-hot-res-type,one-hot-atom-type]
    """
    RES_TYPE_ORIGIN = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    ATOM_TYPE_ORIGIN = ['C','N','O','Others']
    #Open a pdb file
    metas =[]
    with open(input_pdb_file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line[0:4] =="ATOM":
                s = line[54:-1]
                s = s.replace('2CLR','  ')
                metas.append([line[0:6],line[6:11],line[12:16],line[17:20],line[21],line[22:26],line[30:38],line[38:46],line[46:54],s])
    #print(len(metas))
    a = np.array([np.array(x) for x in metas])
    #print(a.shape)
    
    #To DataFrame
    columns = ["atom","atomID","atomName","resName","chainID","resID","x","y","z","others"]
    data = pd.DataFrame(a,columns=columns)
    #print(data.resName.unique())
    data['resName'] = np.where(data['resName']=='MSE','MET', data['resName'])
    #print(data.resName.unique())
    data = data[data.resName.isin(RES_TYPE_ORIGIN)].copy()
    data.index = range(len(data))
    #print(data.shape[0])
    
    #Get chains
    chain_1 = data[data.chainID.isin(chian_id_1)].copy()
    chain_2 = data[data.chainID.isin(chian_id_2)].copy()

    #Covert DataFram as list
    atom_1_list = get_atom_list_from_pdb(chain_1, "R")
    atom_2_list = get_atom_list_from_pdb(chain_2, "L")

    #Get cutoff interface receptor/ligrand atom
    r_list , l_list=get_interface_from_atom_N(atom_1_list,atom_2_list,fix_atom_num//2)

    #Recovery cutoff pdb file
    index = []
    for atom in (r_list):
        index.append(atom[-1])
    for atom in l_list:
        index.append(atom[-1])
    index.sort()
    #print(index)

    cutoff_chain = data.iloc[index,:]
    '''
    with open(sv_pdb_file,'w+') as wf:
        for index, atom in cutoff_chain.iterrows():
            j= atom.values.tolist()
            j = [_.strip() for _ in j]
            j[0] = j[0].ljust(6)#atom#6s
            j[1] = j[1].rjust(5)#aomnum#5d
            j[2] = j[2].center(4)#atomname$#4s
            j[3] = j[3].ljust(3)#resname#1s
            j[4] = j[4].rjust(1) #Astring
            j[5] = j[5].rjust(4) #resnum
            j[6] = str('%8.3f' % (float(j[6]))).rjust(8) #x
            j[7] = str('%8.3f' % (float(j[7]))).rjust(8)#y
            j[8] = str('%8.3f' % (float(j[8]))).rjust(8) #z\
            # j[11]=j[11].rjust(12)#elname    
            wf.write("%s%s %s %s %s%s    %s%s%s %s\n"% (j[0],j[1],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9]))
    '''
    
    '''
    #padding
    r_num = len(r_list)
    l_num = len(l_list)
    for i in range(fix_atom_num//2-r_num):
        r_list.append(r_list[-1])

    for i in range(fix_atom_num//2-l_num):
        l_list.append(l_list[-1])
    '''
    #print(len(r_list))
    #print(len(l_list))
    
    #Encode interfacec atom list
    e_list = encode_atom_list(r_list,l_list,is_atom_list=True)
    
    #padding 0
    r_num = len(r_list)
    l_num = len(l_list)
    line_shape = e_list[0].shape
    r_list_new = e_list[:r_num]
    l_list_new = e_list[r_num:]
    
    for i in range(fix_atom_num//2-r_num):
        r_list_new.append(np.zeros(line_shape))

    for i in range(fix_atom_num//2-l_num):
        l_list_new.append(np.zeros(line_shape))

    return r_list_new+l_list_new

def preprocess_pdb_file_fix_atom(input_pdb_file_path,sv_name,sv_data_folder_path,sv_pdb_folder_path,atom_num,type='txt'):
    """
    Input:
        input_pdb_file: input protein file (.pdb type)
        sv_name: save file name
        sv_data_folder_path: output data file folder (.txt type or .npz type)
        sv_pdb_folder_path: output protein file folder (.pdb type)
        atom_num: fix atom number
        type: 'txt' or 'npz'
    """
    sv_pdb_file_path = os.path.join(sv_pdb_folder_path,sv_name)
    encode_interface_list = process_pdb_file_by_atom_N(input_pdb_file_path,sv_pdb_file_path,['A'],['B'],atom_num)
    point = np.array(encode_interface_list)
    if type=='txt':
        np.savetxt(os.path.join(sv_data_folder_path,sv_name[:-4]+'.txt'),point,delimiter=' ')
    else:
        np.savez(os.path.join(sv_data_folder_path,sv_name[:-4]+'.npz'),point=point)

def single_worker_by_id(pdb_id_list,input_dir,output_dir):
    """
    A process for preprocess pdb file
    An input file folder as:
    
    Input:
        pdb_list: pbd id list for preprocess in this process
        input_dir: pdb dataset input folder
        output_dir:  pdb dataset ouput folder
    """
    for i in trange(len(pdb_id_list)):
        pdb_name = pdb_id_list[i]
        input_path =  os.path.join(input_dir,pdb_name)
        data_output_path = os.path.join(output_dir,'txt',pdb_name)
        pdb_output_path = os.path.join(output_dir,'pdb',pdb_name)
        if not os.path.exists(data_output_path):
            os.mkdir(data_output_path)
        if not os.path.exists(pdb_output_path):
            os.mkdir(pdb_output_path)
        pdb_case_list=[x for x in os.listdir(input_path) if ".pdb" in x]
        for caseid in pdb_case_list:
            input_pdb_path=os.path.join(input_path,caseid)
            preprocess_pdb_file_fix_atom(input_pdb_path,caseid,data_output_path,pdb_output_path,N)

def single_worker_by_file(pdb_file_list,input_dir,output_dir,p_number):
    """
    A process for preprocess pdb file
    An input file folder as:
    
    Input:
        pdb_file_list: pbd file list for preprocess in this process
        input_dir: pdb dataset input folder
        output_dir:  pdb dataset ouput folder
    """
    try:
        for i in trange(len(pdb_file_list)):
            pdb_id,caseid = pdb_file_list[i]
            input_path =  os.path.join(input_dir,pdb_id)
            data_output_path = os.path.join(output_dir,'txt',pdb_id)
            pdb_output_path = os.path.join(output_dir,'pdb',pdb_id)
            input_pdb_path=os.path.join(input_path,caseid)
            preprocess_pdb_file_fix_atom(input_pdb_path,caseid,data_output_path,pdb_output_path,N)
    except Exception as e:
        with open('/home/mxp/chenzihao/error/{}.txt'.format(p_number),'w+') as f:
            print("in No.{} process, exception occurred".format(p_number))
            print("in {}/{}".format(pdb_id,caseid))
            print(str(e))
            f.write("in No.{} process, exception occurred:\n".format(p_number))
            f.write("in {}/{}\n".format(pdb_id,caseid))
            f.write(str(e))

if __name__ == '__main__':
    txt_output_path = os.path.join(output_dir,'txt')
    pdb_output_path = os.path.join(output_dir,'pdb')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(txt_output_path):
        os.mkdir(txt_output_path)
    if not os.path.exists(pdb_output_path):
        os.mkdir(pdb_output_path)

    pdb_id_list = os.listdir(input_dir)
    
    for pdb_id in pdb_id_list:
        data_output_path = os.path.join(output_dir,'txt',pdb_id)
        pdb_output_path = os.path.join(output_dir,'pdb',pdb_id)
        if not os.path.exists(data_output_path):
            os.mkdir(data_output_path)
        if not os.path.exists(pdb_output_path):
            os.mkdir(pdb_output_path)

    num_cores = cpu_count()

    pdb_file_list = []
    for pdb_id in pdb_id_list:
        caseid_list = os.listdir(os.path.join(input_dir,pdb_id))
        for caseid in caseid_list:
            pdb_file_list.append([pdb_id,caseid])
    
    number_pdb_file = len(pdb_file_list)


    p = Pool(processor)
    print("total file numbers: {}".format(number_pdb_file))
    n = number_pdb_file//processor +1
    for i in range(processor):
        start = n*i
        end = number_pdb_file if i==processor-1 else n*(i+1)
        pdb_sub_list = pdb_file_list[start:end]
        print(start,end)
        p.apply_async(single_worker_by_file,args=(pdb_sub_list, input_dir, output_dir,i))
    p.close()
    p.join()
