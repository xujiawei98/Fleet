

import os
import sys

import numpy as np

import paddle
paddle.enable_static()


def save_selectedrows(varname, ids, values, filepath):
    place = paddle.fluid.CPUPlace()
    exe = paddle.fluid.Executor(place)

    def save(var, path):
        prog = paddle.fluid.Program()
        prog.global_block().append_op(
            type='save', inputs={'X': [var]}, outputs={},
            attrs={'file_path': path})
        exe.run(prog)

    var = paddle.fluid.global_scope().var(varname).get_selected_rows()
    var.set_rows(ids)
    var.set_height(len(ids))
    tensor = var.get_tensor()
    tensor.set(np.array(values).astype("float32"), place)
    save(varname, filepath)


def get_distributed_shard(shard_dirname, shard_varname):
    ids = []
    tensors = []

    def get_meta(shard_meta):
        varname = None
        param_dim = -1
        row_names = None
        row_dims = None
        
        with open(shard_meta, "r") as rb:
            for line in rb:
                line = line.strip()
                if line.startswith("param="):
                    varname = line.split("=")[1]
                if line.startswith("row_name"):
                    row_names = line.split("=")[1]
                if line.startswith("row_dims"):
                    row_dims = line.split("=")[1]

            param_dim = row_dims.split(",")[row_names.split(",").index("Param")]
            param_dim = int(param_dim)

        if varname is None or param_dim == -1:
            raise ValueError("can not get right information from {}".format(shard_meta))

        return (varname, param_dim)
        

    def get_embedding(shards, param_dim):
        ids = []
        tensors = []

        for shard in shards:
            with open(shard, "r") as rb:
                for line in rb.readlines():
                    line = line.strip()
                    params = line.split("\t")
                    
                    if len(params) != 5: 
                        raise ValueError("get error value in {}, detail: {}".format(shard, line))

                    feasign = int(params[0])
                    embedding = params[4].split(",")[0:param_dim]
                    embedding = [float(emb) for emb in embedding]
                    ids.append(feasign)
                    tensors.append(embedding)
        return ids, tensors
        

    def get_shard():
        shards = []
        for f in os.listdir(shard_dirname):
            if f.startswith(shard_varname) and f.endswith(".txt"):
                shards.append(os.path.join(shard_dirname, f))
        return shards

    shards = get_shard()

    if len(shards) == 0:
        return None, None

    meta_txt = os.path.join(shard_dirname, "{}.block0.meta".format(shard_varname))        
    meta_varname, param_dim = get_meta(meta_txt)

    if meta_varname != shard_varname:
        raise ValueError("meta error, please check.")

    ids, tensors = get_embedding(shards, param_dim)
    return ids, tensors


if __name__=="__main__":
    if len(sys.argv) != 3:
        raise ValueError("only accept model dirname, varname, python convert.py output/epoch_0 embedding")

    dirname = sys.argv[1]
    varname = sys.argv[2]

    shard_txt = os.path.join(dirname, "{}_txt".format(varname))
    selected_rows = os.path.join(dirname, varname)

    if not os.path.exists(shard_txt):
        raise ValueError("{} is not exist, pleast confirm your argv.".format(shard_txt))

    if os.path.exists(selected_rows):
        raise ValueError("{} is exist, pleast delete.".format(selected_rows))

    print("searching Param/Meta from {} and will merge to {}".format(shard_txt, selected_rows))

    ids, tensors = get_distributed_shard(shard_txt, varname)
    save_path = os.path.join(dirname, varname)
    save_selectedrows(varname, ids, tensors, save_path)

    print("save {} with ids: {} to {}".format(varname, len(ids), save_path))

    
