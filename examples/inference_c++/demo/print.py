import paddle
#paddle.enable_static()

from paddle.fluid.framework import Program

model_basename = "__model__"
model_filename = model_basename

with open(model_filename, "rb") as f:
    program_desc_str = f.read()

program = Program.parse_from_string(program_desc_str)
print(program)

