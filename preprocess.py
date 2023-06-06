"""
Usage:
python preprocess.py --p_num {process_number} --npoint {N} --dataset_dir {dataset_dir} --data_sv_dir {data_save_dir}
"""
import os
import sys
import argparse
from re import L
import numpy as np
import operator
from functools import reduce
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import trange
import utils.provider as provider
RESIDUE_Forbidden_SET={"FAD"}
CUT_OFF=8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR,'utils'))


MED_OUT_PATH = os.path.join(ROOT_DIR,'dockground61_interface')
if not os.path.exists(MED_OUT_PATH):
    os.mkdir(MED_OUT_PATH)



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('pre-processing')
    parser.add_argument('--p_num', type=int, default=20, help='processor number [default: 24]')
    parser.add_argument('--npoint', type=int, default=1000, help='Point cloud total number [default: 1000]')
    parser.add_argument('--dataset_dir', type=str, default='dockground61', help='Dataset dir [format is shown in README]')
    parser.add_argument('--data_folder', type=str, default='data', help='Data dir [default: data]')
    parser.add_argument('--data_sv_dir', type=str, default='dockground61_1000', help='Special Data save dir')
    parser.add_argument('--error_dir', type=str, default='error', help='Error message save dir')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--cutoff', type=int, default=8, help='cutoff') #todo
    parser.add_argument('--seed', type=int, default=1024, help='random seed') 

    return parser.parse_args()

def Write_Interface(line_list,pdb_path,ext_file):
    new_path=pdb_path[:-4]+ext_file
    with open(new_path,'w') as file:
        for line in line_list:
            #check residue in the common residue or not. If not, no write for this residue
            residue_type = line[17:20]
            if residue_type in RESIDUE_Forbidden_SET:
                continue
            file.write(line)
    return new_path
    

def Extract_Interface(pdb_path):
    """
    Input:
        pdb_path: pdb_path
    Return:
        final_receptor: list of all recptor atom lines
        final_ligand: list of all ligand atom lines
    """
    receptor_list=[]
    ligand_list=[]
    rlist=[]
    llist=[]
    count_r=0
    count_l=0
    with open(pdb_path,'r') as file:
        line = file.readline()               # call readline()
        while line[0:4]!='ATOM':
            line=file.readline()
        atomid = 0
        count = 1
        goon = False
        chain_id = line[21]
        first_chain_id = chain_id
        residue_type = line[17:20]
        pre_residue_type = residue_type
        tmp_list = []
        pre_residue_id = 0
        pre_chain_id = line[21]
        first_change=True
        while line:

            dat_in = line[0:80].split()
            if len(dat_in) == 0:
                line = file.readline()
                continue

            if (dat_in[0] == 'ATOM'):
                chain_id = line[21]
                residue_id = int(line[23:26])

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_type = line[17:20]
                # First try CA distance of contact map
                atom_type = line[13:16].strip()
                if chain_id!=first_chain_id:
                    goon=True
                if (goon):
                    if first_change:
                        rlist.append(tmp_list)
                        tmp_list = []
                        tmp_list.append([x, y, z, atom_type, count_l])
                        count_l += 1
                        ligand_list.append(line)
                        first_change=False
                    else:
                        ligand_list.append(line)  # used to prepare write interface region
                        if pre_residue_type == residue_type:
                            tmp_list.append([x, y, z, atom_type, count_l])
                        else:
                            llist.append(tmp_list)
                            tmp_list = []
                            tmp_list.append([x, y, z, atom_type, count_l])
                        count_l += 1
                else:
                    receptor_list.append(line)
                    if pre_residue_type == residue_type:
                        tmp_list.append([x, y, z, atom_type, count_r])
                    else:
                        rlist.append(tmp_list)
                        tmp_list = []
                        tmp_list.append([x, y, z, atom_type, count_r])
                    count_r += 1

                atomid = int(dat_in[1])
                chain_id = line[21]
                count = count + 1
                pre_residue_type = residue_type
                pre_residue_id = residue_id
                pre_chain_id = chain_id
            line = file.readline()
    print("Extracting %d/%d atoms for receptor, %d/%d atoms for ligand"%(len(receptor_list),count_r,len(ligand_list),count_l))
    final_receptor, final_ligand=Form_interface(rlist,llist,receptor_list,ligand_list)

    return final_receptor,final_ligand

def Form_interface(rlist,llist,receptor_list,ligand_list,cut_off=CUT_OFF):
    """
    Input:
        rlist: list of all recptor residual
        llist: list of all recptor residual
        receptor_list: list of all recptor atom lines
        ligand_list: list of all recptor atom lines
        cut_off: cut off disatance (default: 10 A)
    Return:
        final_receptor: list of all cut off recptor atom lines
        final_ligand: list of all cut off ligand atom lines
    """
    cut_off=cut_off**2
    r_index=set()
    l_index=set()
    for rindex,item1 in enumerate(rlist):
        for lindex,item2 in enumerate(llist):
            min_distance=1000000
            residue1_len=len(item1)
            residue2_len=len(item2)
            for m in range(residue1_len):
                atom1=item1[m]
                for n in range(residue2_len):
                    atom2=item2[n]
                    distance=0
                    for k in range(3):
                        distance+=(atom1[k]-atom2[k])**2
                    #distance=np.linalg.norm(atom1[:3]-atom2[:3])
                    if distance<=min_distance:
                        min_distance=distance
            if min_distance<=cut_off:
                if rindex not in r_index:
                    r_index.add(rindex)
                if lindex not in l_index:
                    l_index.add(lindex)
    r_index=list(r_index)
    l_index=list(l_index)
    newrlist=[]
    for k in range(len(r_index)):
        newrlist.append(rlist[r_index[k]])
    newllist=[]
    for k in range(len(l_index)):
        newllist.append(llist[l_index[k]])
    print("After filtering the interface region, %d/%d residue in receptor, %d/%d residue in ligand" % (len(newrlist),len(rlist), len(newllist),len(llist)))
    #get the line to write new interface file
    final_receptor=[]
    final_ligand=[]
    for residue in newrlist:
        for tmp_atom in residue:
            our_index=tmp_atom[4]
            final_receptor.append(receptor_list[our_index])

    for residue in newllist:
        for tmp_atom in residue:
            our_index=tmp_atom[4]
            #print (our_index)
            final_ligand.append(ligand_list[our_index])
    print("After filtering the interface region, %d receptor, %d ligand"%(len(final_receptor),len(final_ligand)))

    return final_receptor,final_ligand

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
        res_name = str(_atom[3]).strip()
        res_id = str(_atom[5]).strip()
        x = float(str(_atom[6]).strip())
        y = float(str(_atom[7]).strip())
        z = float(str(_atom[8]).strip())
        #print(x,y,z)
        if atom_name not in ATOM_TYPE:
            atom_name = 'Others'
        atom_tuple = [x,y,z,atom_name,res_name,res_id,chain_type,index]
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

def process_pdb_file_by_atom_N(input_pdb_file,sv_pdb_file,chian_id_1='A',chian_id_2='B',npoint=1000,med_out=True):
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
    #Open a pdb file
    final_receptor,final_ligand = Extract_Interface(input_pdb_file)

    if med_out:
        pdb_id = input_pdb_file.split(os.path.sep)[-2]
        pdb_name = input_pdb_file.split(os.path.sep)[-1]
        if not os.path.exists(os.path.join(MED_OUT_PATH,pdb_id)):
            os.mkdir(os.path.join(MED_OUT_PATH,pdb_id))
        Write_Interface(final_receptor+final_ligand,os.path.join(MED_OUT_PATH,pdb_id,pdb_name),".pdb")

    r_atom_list = []
    l_atom_list = []
    for line in final_receptor:
        line = line.strip()
        if line[0:4] =="ATOM":
            sep = line[54:-1]
            sep = sep.replace('2CLR','  ')
            r_atom_list.append([line[0:6],line[6:11],line[12:16],line[17:20],line[21],line[22:26],line[30:38],line[38:46],line[46:54],sep])

    for line in final_ligand:
        line = line.strip()
        if line[0:4] =="ATOM":
            sep = line[54:-1]
            sep = sep.replace('2CLR','  ')
            l_atom_list.append([line[0:6],line[6:11],line[12:16],line[17:20],line[21],line[22:26],line[30:38],line[38:46],line[46:54],sep])
    
    #To DataFrame
    columns = ["atom","atom_id","atom_name","res_name","chain_id","res_id","x","y","z","others"]
    data_r = pd.DataFrame(r_atom_list,columns=columns)
    data_l = pd.DataFrame(l_atom_list,columns=columns)
    
    #print(data.resName.unique())
    data_r['res_name'] = np.where(data_r['res_name']=='MSE','MET', data_r['res_name'])
    data_l['res_name'] = np.where(data_l['res_name']=='MSE','MET', data_l['res_name'])
    #print(data.resName.unique())

    data_r = data_r[data_r['res_name'].isin(RES_TYPE_ORIGIN)].copy()
    data_r.index = range(len(data_r))

    data_l = data_l[data_l['res_name'].isin(RES_TYPE_ORIGIN)].copy()
    data_l.index = range(len(data_l))

    #Covert DataFram as list
    atom_r_list = get_atom_list_from_pdb(data_r, "R")
    atom_l_list = get_atom_list_from_pdb(data_l, "L")

    #Get cutoff interface receptor/ligrand atom
    fix_r_list , fix_l_list = get_interface_from_atom_N(atom_r_list,atom_l_list,npoint//2)

    if med_out:
        pass

    e_list = encode_atom_list(fix_r_list,fix_l_list,is_atom_list=True)
    
    #padding 0
    r_num = len(fix_r_list)
    l_num = len(fix_l_list)
    line_shape = e_list[0].shape
    r_list_encode = e_list[:r_num]
    l_list_encode = e_list[r_num:]
    
    for i in range(npoint//2-r_num):
        r_list_encode.append(np.zeros(line_shape))

    for i in range(npoint//2-l_num):
        l_list_encode.append(np.zeros(line_shape))

    return r_list_encode+l_list_encode

def preprocess_one_pdb_file(input_pdb_file_path,sv_name,sv_data_folder_path,npoint,type='txt'):
    """
    Input:
        input_pdb_file: input protein file (.pdb type)
        sv_name: save file name
        sv_data_folder_path: output data file folder (.txt type or .npz type)
        atom_num: fix atom number
        type: 'txt' or 'npz'
    """
    sv_pdb_file_path = os.path.join(sv_data_folder_path,sv_name)
    output_point_cloud_data = process_pdb_file_by_atom_N(input_pdb_file_path,sv_pdb_file_path,['A'],['B'],npoint)
    point = np.array(output_point_cloud_data)
    if type=='txt':
        np.savetxt(os.path.join(sv_data_folder_path,sv_name[:-4]+'.txt'),point,delimiter=' ')
    else:
        np.savez(os.path.join(sv_data_folder_path,sv_name[:-4]+'.npz'),point=point)

def single_worker_by_pdb_id_list(pdb_id_list,input_dir,output_dir,npoint):
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
            preprocess_one_pdb_file(input_pdb_path,caseid,data_output_path,pdb_output_path,npoint)

def single_worker_by_file_list(pdb_file_list,input_dir,output_dir,error_dir,p_number,npoint):
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
            input_pdb_path= os.path.join(input_path,caseid)
            output_path = os.path.join(output_dir,pdb_id)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            preprocess_one_pdb_file(input_pdb_path,caseid,output_path,npoint)
    except Exception as e:
        with open(os.path.join(error_dir,'p_{}.txt'.format(p_number)),'w+') as f:
            print("in No.{} process, exception occurred".format(p_number))
            print("in {}/{}".format(pdb_id,caseid))
            print(str(e))
            f.write("in No.{} process, exception occurred:\n".format(p_number))
            f.write("in {}/{}\n".format(pdb_id,caseid))
            f.write(str(e))

def main():
    args = parse_args()
    provider.set_seed(args.seed)

    DATASET_PATH = os.path.join(ROOT_DIR,args.dataset_dir)
    DATA_PATH = os.path.join(ROOT_DIR,args.data_folder)
    ERROR_PATH = os.path.join(ROOT_DIR,args.error_dir)

    if not os.path.exists(ERROR_PATH):
        os.mkdir(ERROR_PATH)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    DATA_SV_PATH = os.path.join(DATA_PATH,args.data_sv_dir)
    if not os.path.exists(DATA_SV_PATH):
        os.mkdir(DATA_SV_PATH)


    pdb_id_list = os.listdir(DATASET_PATH)
    pdb_file_list = []
    for pdb_id in pdb_id_list:
        caseid_list = os.listdir(os.path.join(DATASET_PATH,pdb_id))
        caseid_list = [x for x in caseid_list if x.endswith('.pdb')]
        for caseid in caseid_list:
            pdb_file_list.append([pdb_id,caseid])
    
    number_pdb_file = len(pdb_file_list)
    print("total file numbers: {}".format(number_pdb_file))


    process_pool = Pool(args.p_num)
    n = number_pdb_file//args.p_num +1
    for p_id in range(args.p_num):
        start = n*p_id
        end = number_pdb_file if p_id==args.p_num-1 else n*(p_id+1)
        pdb_sub_list = pdb_file_list[start:end]
        process_pool.apply_async(single_worker_by_file_list,args=(pdb_sub_list, DATASET_PATH, DATA_SV_PATH, ERROR_PATH, p_id, args.npoint))
    process_pool.close()
    process_pool.join()


if __name__ == '__main__':
    main()
