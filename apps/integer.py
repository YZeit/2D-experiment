import streamlit as st
from docplex.mp.sdetails import SolveDetails
import matplotlib.pyplot as plt

def app():
    st.header('Bi-objective experiment')
    #example bi-objective linear program
    st.subheader('Example:')

    st.latex(r'''
        \begin{array}{rll}
         \min & f_1(x) = 3x1 + x2 & \\
         & f_2(x) = -x1 + 4x2 & \\
         s.t.: & x_2 \leqslant 6 & \\
         & x1 + 3x2 \geqslant 3 & \\
         & 2x1 - x2 \leqslant 6 & \\
         & 2x1 + x2 \leqslant 10 & \\
         & x1, x2 \geqslant 0  & \\
         & x1, x2 \in \mathbb{Z}  & \\
    \end{array}
    ''')

    reference_point = [0,0]

    reference_point[0] = st.sidebar.slider("Please select the reference point for the 1st objective", min_value=-10.00, max_value=25.00, value=0.00)
    reference_point[1] = st.sidebar.slider("Please select the reference point for the 2nd objective", min_value=-10.00, max_value=25.00, value=0.00)

    weighting_normalized = [0.5,0.5]
    #weighting_normalized[0] = st.sidebar.slider('Please select the weighting for the 1st objective', min_value=0.0, max_value=1.0, value=0.5)
    #weighting_normalized[1] = st.sidebar.slider('Please select the weighting for the 2nd objective', min_value=0.0, max_value=1.0, value=1-weighting_normalized[0])

    #a_f1 = 0.6
    #a_f2 = 0.3
    #a_f1f2 = 0.1

    f1_nadir = 14
    f2_nadir = 24

    f1_utopian = 1
    f2_utopian = -3

    # Optimization Model
    from docplex.mp.model import Model
    m = Model(name='Bi-objective convex Pareto Front')

    roh = 0.000001  # arbitrary positive small value

    # decision variables
    x1 = m.integer_var(lb=0)
    x2 = m.integer_var(lb=0)
    nu = m.continuous_var(name='nu')

    # constraints
    constraint1 = x2 <= 6
    constraint2 = x1 + 3 * x2 >= 3
    constraint3 = 2 * x1 - x2 <= 6
    constraint4 = 2 * x1 + x2 <= 10
    m.add_constraint(constraint1)
    m.add_constraint(constraint2)
    m.add_constraint(constraint3)
    m.add_constraint(constraint4)

    # objective functions
    objective = [0, 0]
    objective[0] = 3 * x1 + x2
    objective[1] = -x1 + 4 * x2

    feasible_solutions_z1 = []
    feasible_solutions_z2 = []

    for i in range(1,15):
        m.add_constraint(objective[0] == i, ctname='temp1')
        for i in range(-3, 25):
            m.add_constraint(objective[1] == i, ctname='temp2')
            try:
                m.minimize(objective[0])
                m.solve()
                # st.write('iteration: ' + str(i))
                # st.write(objective[0].solution_value)
                feasible_solutions_z1.append(objective[0].solution_value)
                # st.write(objective[1].solution_value)
                feasible_solutions_z2.append(objective[1].solution_value)
                m.remove_constraint('temp2')
            except:
                m.remove_constraint('temp2')
        m.remove_constraint('temp1')




    #st.write(feasible_solutions_z1)
    #st.write(feasible_solutions_z2)


    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.header('The feasible region')
    #extreme points x1 and x2
    #extreme_points_x1 = [0, 0, 3, 4, 2, 0]
    #extreme_points_x2 = [6, 1, 0, 2, 6, 6]
    #plt.plot(extreme_points_x1, extreme_points_x2)
    #plt.xlabel("$x_1$")
    #plt.ylabel("$x_2$")
    #st.pyplot()
    c1, c2 = st.beta_columns((1, 1))
    with c1:
        st.header('Achievement Scalarizing Function')
        st.latex(r'''    {\displaystyle \sigma(z,z^r,\lambda,\rho) = \max_{g=1,2,\ldots,\ell}{\lambda_g(z_g-z_g^r)} + \rho \sum_{g=1}^\ell\lambda_g(z_g-z_g^r),}''')

        st.latex(r'''
            \begin{array}{rll}
             \min & {\nu + \rho \sum_{g=1}^\ell \lambda_g (z_g - z_g^r)} & \\
             s.t.: & \lambda_g (z_g - z_g^r) \leqslant \nu, & g = 1,2 \\
             & x \in X & \\
             & \nu \in \mathbb{R}. & \\
        \end{array}
        ''')

    with c2:
        #Choquet
        st.header('Choquet Integral')
        st.write(
            '$C_\mu(x)=\sum_{\{i\}\subseteq G} a(\{i\})(z_i(x)) + \sum_{\{i,j\}\subseteq G} a(\{i,j\})\min(z_i(x),z_j(x))$')
        st.latex(r'''{\displaystyle \Delta(x)=\frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )},\ldots,\frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )},\ldots,\frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad})}}''')
        st.latex(r'''\min_{x \in X} C_\mu \big(\Delta(x)\big)''')
        st.latex(r'''\begin{array}{rcl}
        {\displaystyle C_\mu(\Delta x)}  & = & {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\})\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right) \; +} \\  
        & + &  {\displaystyle\sum_{\{i,j\}\subseteq G} a(\{i,j\})\min\left\{\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right),\; \left(\frac{z_j^r-z_j (x)}{z_j^r-z_j^{nad}}\right)\right\}},
    \end{array} ''')
        st.write('''To linearize the problem, let's introduce a new variable $u$:''')
        st.latex(r'''{\displaystyle u=\min \big(\frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )},\ldots,\frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )},\ldots,\frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad})}}\big)''')
        st.latex(r'''\begin{array}{l}
        {\displaystyle u \geqslant \frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )} - My_1} \\
        {\displaystyle ...} \\
        {\displaystyle u \geqslant \frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )} - My_k} \\
            ... \\
        {\displaystyle u \geqslant \frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad} )} - My_p} \\
        {\displaystyle \sum_{k=1}^p y_k = p-1}
        \end{array}''')
        st.latex(r'''\min_{x \in X}\max_{k=1,2,\ldots,p} \left\{C_\mu \big(\Delta(x_k)\big)\right\},''')
        st.latex(r'''    \begin{array} {rcl}
    {\displaystyle C_\mu \big(\Delta(x_k)\big)} & = & {\displaystyle(a(\{k\})\left(\frac{z_k^r-z_k (x)}{|z_k^r-z_k^{nad}|}\right)+} \\ 
    & + & {\displaystyle\sum_{\{k,j\}\subseteq G} a(\{k,j\})\min\left\{\left(\frac{z_k^r-z_k (x)}{|z_k^r-z_k^{nad}|}\right), \left(\frac{z_j^r-z_j (x)}{|z_j^r-z_j^{nad}|}\right)\right\}}
    \end{array}''')
        st.latex(r'''    \begin{array}{rll}
         \min & \nu  & \\
         s.t.: & {\displaystyle C_\mu \big(\Delta(x_k)\big)} \leqslant \nu, & k = 1,2 \\
         & x \in X & \\
         & \nu \in \mathbb{R}. & \\
    \end{array}''')
    c1, c2 = st.beta_columns((1, 1))
    with c1:
        st.subheader('Weightings Assumptions')
        # weighting_normalized[0] = st.slider('Please select the weighting for the 1st objective', min_value=0.0, max_value=1.0, value=0.5)
        # weighting_normalized[1] = st.slider('Please select the weighting for the 2nd objective', min_value=0.0, max_value=1.0, value=1-weighting_normalized[0])
        st.write('$\lambda_1$')
        weighting_normalized[0] = st.slider('Please select the weighting for the 1st objective', min_value=0.0,
                                            max_value=1.0, value=0.5)
        st.write('$\lambda_2$')
        weighting_normalized[1] = st.slider('Please select the weighting for the 2nd objective', min_value=0.0,
                                            max_value=1.0, value=1 - weighting_normalized[0])
    with c2:
        st.subheader('Preference Assumptions:')
        st.write('$a(\{z_1\})$')
        a_f1 = st.slider('Please select the value for the 1st criterion', min_value=0.00, max_value=2.00,
                         value=0.60)
        st.write('$a(\{z_2\})$')
        a_f2 = st.slider('Please select the value for the 2nd criterion', min_value=0.00, max_value=2.00,
                         value=0.30)
        st.write('$a(\{z_1,z_2\})$')
        a_f1f2 = st.slider('Please select the value for interaction of both criteria', min_value=-1.00,
                           max_value=1.00, value=0.10)

    # achievement scalarizing function
    m.add_constraint((weighting_normalized[0] * (objective[0] - reference_point[0]) <= nu), ctname='nu1')
    m.add_constraint((weighting_normalized[1] * (objective[1] - reference_point[1]) <= nu), ctname='nu2')
    achievement_scalarizing_function = nu + roh * sum(weighting_normalized[i] * (objective[i] - reference_point[i]) for i in range(2))

    m.minimize(achievement_scalarizing_function)
    m.solve()
    solve_details = m.solve_details
    # st.write(solve_details)
    x1_solution_achievement = x1.solution_value
    x2_solution_achievement = x2.solution_value
    z1_solution_achievement = objective[0].solution_value
    z2_solution_achievement = objective[1].solution_value

    bigm = 100000
    #y = m.binary_var()
    y_1 = m.binary_var()
    y_2 = m.binary_var()
    min_value = m.continuous_var()
    m.remove_constraint('nu1')
    m.remove_constraint('nu2')
    nu_1=m.continuous_var()


    #delta_x1 = m.continuous_var()
    #delta_x2 = m.continuous_var()
    delta_x1 = ((reference_point[0]-objective[0])/-abs(reference_point[0]-f1_nadir))
    delta_x2 = ((reference_point[1]-objective[1])/-abs(reference_point[1]-f2_nadir))
    m.add_constraint(min_value >= delta_x1-(bigm*y_1))
    m.add_constraint(min_value >= delta_x2-(bigm*y_2))
    #m.add_constraint(min_value <= delta_x1+(bigm*y_1))
    #m.add_constraint(min_value <= delta_x2+(bigm*y_2))
    m.add_constraint(y_1+y_2==1)

    m.add_constraint(a_f1 * delta_x1 + (a_f1f2 * min_value) <= nu_1)
    m.add_constraint(a_f2 * delta_x2 + (a_f1f2 * min_value) <= nu_1)
    #m.add_constraint(delta_x1-delta_x2<=bigm*(1-y))
    #m.add_constraint(delta_x2-delta_x1<=bigm*y)
    #m.add_constraint(delta_x1 >= min_value)
    #m.add_constraint(delta_x2 >= min_value)
    #m.add_constraint(delta_x1 - bigm*(1-y) <= min_value)
    #m.add_constraint(delta_x2 - bigm*y <= min_value)
    #delta_x1_plus = m.continuous_var(lb=0)
    #delta_x1_minus = m.continuous_var(lb=0)
    #delta_x2_plus = m.continuous_var(lb=0)
    #delta_x2_minus = m.continuous_var(lb=0)
    #m.add_constraint(delta_x1==delta_x1_plus-delta_x1_minus)
    #m.add_constraint(delta_x2 == delta_x2_plus-delta_x2_minus)
    #m.add_constraint(delta_x1_plus-delta_x1_minus-delta_x2_plus+delta_x2_minus <= bigm*(1-y))
    #m.add_constraint(delta_x2_plus-delta_x2_minus-delta_x1_plus+delta_x1_minus <= bigm*y)
    #m.add_constraint(delta_x1_plus-delta_x1_minus >= min_value)
    #m.add_constraint(delta_x2_plus-delta_x2_minus >= min_value)
    #m.add_constraint(delta_x1_plus-delta_x1_minus - bigm*(1-y) <= min_value)
    #m.add_constraint(delta_x2_plus-delta_x2_minus - bigm*y <= min_value)
    #alpha_1 = m.binary_var()
    #alpha_2 = m.binary_var()
    #alpha = m.binary_var()
    #m.add_constraint(delta_x1_minus-delta_x1_plus <= bigm*alpha_1)
    #m.add_constraint(delta_x1_plus-delta_x1_minus <= bigm*(1-alpha_1))
    #m.add_constraint(delta_x2_minus-delta_x2_plus <= bigm*alpha_2)
    #m.add_constraint(delta_x2_plus-delta_x2_minus <= bigm*(1-alpha_2))
    #m.add_constraint(alpha_1 <= alpha)
    #m.add_constraint(alpha_2 <= alpha)


    #m.add_constraint(objective[0]<=20)
    choquet = a_f1*delta_x1 + a_f2*delta_x2+ (a_f1f2*min_value)
    #m.add_constraint(a_f1*((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir)) + a_f2*((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir)) + a_f1f2*min_value <= nu)
    #m.add_constraint(a_f1*((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir)) <= nu)
    #m.add_constraint(a_f2*((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir)) <= nu)
    m.minimize(nu_1)
    m.solve()
    solve_details = m.solve_details
    #st.write(solve_details)


    x1_solution_choquet = x1.solution_value
    x2_solution_choquet = x2.solution_value
    z1_solution_choquet = objective[0].solution_value
    z2_solution_choquet = objective[1].solution_value


    c1, c2 = st.beta_columns((1, 1))
    with c1:
        st.subheader('Solution')
        st.write('$x_1$: ' + str(x1_solution_achievement))
        st.write('$x_2$: ' + str(x2_solution_achievement))
        st.write('$z_1(x)$: ' + str(z1_solution_achievement))
        st.write('$z_2(x)$: ' + str(z2_solution_achievement))
    with c2:
        st.subheader('Solution')
        st.write('$x_1$: ' + str(x1_solution_choquet))
        st.write('$x_2$: ' + str(x2_solution_choquet))
        st.write('$z_1(x)$: ' + str(z1_solution_choquet))
        st.write('$z_2(x)$: ' + str(z2_solution_choquet))
        st.write('min value: ' + str(min_value.solution_value))
        #st.write('delta x_1: ' + str(delta_x1.solution_value))
        #st.write('delta x_2: ' + str(delta_x2.solution_value))
        st.write('choquet: ' + str(choquet.solution_value))

    c1, c2 = st.beta_columns((1, 1))
    with c1:
        st.header('The projected feasible region')
        fig2, ax1 = plt.subplots()
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.yticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        plt.xticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        plt.grid(True)
        #ax1 = fig2.add_subplot()
        ax1.set_xlim([-10, 25])
        ax1.set_ylim([-10, 25])
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.tick_params(axis='both', which='minor', labelsize=4)
        extreme_points_z1 = [6, 1, 9, 14, 12, 6]
        extreme_points_z2 = [24, 4, -3, 4, 22, 24]
        ax1.plot(extreme_points_z1, extreme_points_z2, dashes=[6, 2], linewidth=1, label='feasible area')
        ax1.scatter(feasible_solutions_z1, feasible_solutions_z2, s=6, c='b', label='feasible solutions')
        ax1.scatter(reference_point[0], reference_point[1], color='y', label='reference point')
        ax1.scatter(z1_solution_achievement, z2_solution_achievement,color='g', label='obtained solution')
        ax1.plot([reference_point[0], z1_solution_achievement], [reference_point[1], z2_solution_achievement])
        ax1.legend()
        st.pyplot(fig2)
    with c2:
        st.header('The projected feasible region')
        fig2, ax1 = plt.subplots()
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.yticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        plt.xticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        plt.grid(True)
        #ax1 = fig2.add_subplot()
        ax1.set_xlim([-10, 25])
        ax1.set_ylim([-10, 25])
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.tick_params(axis='both', which='minor', labelsize=4)
        extreme_points_z1 = [6, 1, 9, 14, 12, 6]
        extreme_points_z2 = [24, 4,-3, 4, 22, 24]
        ax1.plot(extreme_points_z1, extreme_points_z2, dashes=[6, 2], linewidth=1, label='feasible area')
        ax1.scatter(feasible_solutions_z1, feasible_solutions_z2, s=6, c='b', label='feasible solutions')
        ax1.scatter(reference_point[0], reference_point[1], color='y', label='reference point')
        ax1.scatter(z1_solution_choquet, z2_solution_choquet, color='g', label='obtained solution')
        ax1.plot([reference_point[0], z1_solution_choquet], [reference_point[1], z2_solution_choquet])
        ax1.legend()
        st.pyplot(fig2)



