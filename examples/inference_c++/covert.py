import os
import shutil
import numpy as np

import paddle
paddle.enable_static()

from paddle.fluid.framework import Program
import paddle.fluid.core as core


def create_var_struct(var):
    if var.type == core.VarDesc.VarType.SELECTED_ROWS:
        lod_level = None
    elif var.type == core.VarDesc.VarType.LOD_TENSOR:
        lod_level = var.lod_level
    else:
        raise ValueError("can only support SELECTED_ROWS/LOD_TENSOR now")

    return VarStruct(var.name, var.shape, var.dtype, var.type, lod_level,
                     var.persistable)


class VarStruct(object):
    """
    record part properties of a Variable in python.
    """

    def __init__(self, name, shape, dtype, type, lod_level, persistable):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.type = type
        self.lod_level = lod_level
        self.persistable = persistable

    def remove_var(self, block):
        if program.global_block().has_var(self.name):
            program.global_block()._remove_var(self.name)

    def create_selectedrows_var(self, block):
        block.create_var(name="{}".format(self.name),
                         dtype=self.dtype,
                         persistable=self.persistable,
                         type=core.VarDesc.VarType.SELECTED_ROWS,
                         shape=self.shape)

    def __str__(self):
        return "N: {}, S: {}, D: {}, T: {}, LL: {}, P: {}".format(
            self.name, self.shape, self.dtype, self.type, self.lod_level,
            self.persistable)


modir = "output/trainer_0_epoch_0/"
param_dim = 10

model_basename = "__model__"
model_file = os.path.join(modir, model_basename)
backup_model_file = os.path.join(modir, "{}.old".format(model_basename))

#if os.path.exists(backup_model_file):
#    raise ValueError("{} existed, please check".format(backup_model_file))

shutil.move(model_file, backup_model_file)

with open(backup_model_file, "rb") as f:
    program_desc_str = f.read()

program = Program.parse_from_string(program_desc_str)
sparse_tables = set()

for op in program.global_block().ops:
    if op.type == "lookup_table" or op.type == "lookup_table_v2":
        sparse_tables.add(op.input("W")[0])

sparse_tables = list(sparse_tables)
varmetas = [create_var_struct(program.global_block().vars[table]) for table in sparse_tables]

for meta in varmetas:
    meta.remove_var(program.global_block())
    meta.create_selectedrows_var(program.global_block())

with open(model_file, "wb") as f:
    f.write(program.desc.serialize_to_string())

print("covert __model__ done")

for table in sparse_tables:
    path = os.path.join(modir, table)
    if not os.path.isdir(path):
        raise ValueError("There is no directory named '%s'", path)


def save(var, path):
    place = paddle.fluid.CPUPlace()
    exe = paddle.fluid.Executor(place)
    prog = paddle.fluid.Program()
    prog.global_block().append_op(
        type='save', inputs={'X': [var]}, outputs={},
        attrs={'file_path': path})
    exe.run(prog)

ids = []
values = []
shutil.move(os.path.join(modir, table), os.path.join(modir, "{}.shard".format(table)))

for table in sparse_tables:
    path = os.path.join(modir, "{}.shard".format(table))
    for f in os.listdir(path):
        if not f.startswith(table) or not f.endswith(".txt"):
            continue
        abs_file = os.path.join(path, f)
        with open(abs_file, "r") as rb:
            for l in rb.readlines():
                l = l.strip()
                id, params, _, _, _, _, _ = l.split("\t")
                ids.append(int(id))
                values.append([float(p) for p in params.split(",")])

    var = paddle.fluid.global_scope().var(table).get_selected_rows()
    var.set_rows(ids)
    var.set_height(len(ids))
    tensor = var.get_tensor()
    place = paddle.fluid.CPUPlace()
    tensor.set(np.array(values), place)
    save(table, "{}/{}".format(modir, table))

