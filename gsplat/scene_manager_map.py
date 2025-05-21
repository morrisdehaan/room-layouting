file_path = '/home/scur0703/.conda/envs/gsplat_env_test/lib/python3.11/site-packages/pycolmap/scene_manager.py'

new_lines=[]
with open(file_path, 'r') as f:
  for lines in f:
    # print(lines.strip())
    file_line = lines
    if "append(map(" in file_line:
      file_line = file_line.replace("(map", "(np.array(list(map")
      # file_line = file_line.replace("np.array(list(map(", "np.array(map(")
      # file_line = file_line.replace("self.point3D_colors.append(np.array(list(map((np.uint8, data[4:7]))))", "self.point3D_colors.append(np.array(list(map(np.uint8, data[4:7]))))")
      # file_line = file_line.replace(")))", "))))")
      file_line = file_line.replace("))", "))))")
    new_lines.append(file_line)

# print(new_lines)
with open(file_path, 'w') as file:
    file.writelines(new_lines)