import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jmx001/my_program/my_robot_project/install/peg_in_hole'
