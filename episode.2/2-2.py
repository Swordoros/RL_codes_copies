#求解示例bellman最优方程

import sympy
from sympy import symbols
import warnings
warnings.filterwarnings("ignore")


sympy.init_printing()
alpha, beta, gamma = symbols('alpha beta gamma')
v_hungry, v_full= symbols('v_hungry v_full')
q_hungry_eat, q_hungry_none, q_full_eat, q_full_none = symbols('q_hungry_eat q_hungry_none q_full_eat q_full_none')
xy_tuples=((0,0),(1,0),(0,1),(1,1))
for x,y in xy_tuples:
    system = sympy.Matrix((
        (1,0,x-1,-x,0,0,0),
        (0,1,0,0,-y,y-1,0),
        (-gamma,0,1,0,0,0,-2),
        ((alpha-1)*gamma, -alpha*gamma, 0, 1, 0, 0, 4*alpha-3),
        (-beta*gamma, (beta-1)*gamma, 0, 0, 1, 0, -4*beta+2),
        (0,-gamma,0,0,0,1,1)
    ))
    result = sympy.solve_linear_system(system, v_hungry, v_full, q_hungry_none, q_hungry_eat, q_full_none, q_full_eat)
    msgx ='v(饿) = q(饿, {}吃)'.format(''if x else '不')
    msgy ='v(饱) = q(饱, {}吃)'.format('不'if y else '')
    print('==== {}, {} ==== x={}, y={} ===='.format(msgx, msgy, x, y))
    print(result)